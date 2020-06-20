use imgui::{
    Context, DrawCmd::Elements, DrawData, DrawIdx, DrawList, DrawVert, TextureId, Textures,
};
use std::mem::size_of;
use wgpu::*;

pub type RendererResult<T> = Result<T, RendererError>;

#[derive(Clone, Debug)]
pub enum RendererError {
    BadTexture(TextureId),
}

#[allow(dead_code)]
enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

#[cfg(feature = "glsl-to-spirv")]
struct Shaders;

#[cfg(feature = "glsl-to-spirv")]
impl Shaders {
    fn compile_glsl(code: &str, stage: ShaderStage) -> Vec<u32> {
        let ty = match stage {
            ShaderStage::Vertex => glsl_to_spirv::ShaderType::Vertex,
            ShaderStage::Fragment => glsl_to_spirv::ShaderType::Fragment,
            ShaderStage::Compute => glsl_to_spirv::ShaderType::Compute,
        };

        read_spirv(glsl_to_spirv::compile(&code, ty).unwrap()).unwrap()
    }

    fn get_program_code() -> (&'static str, &'static str) {
        (include_str!("imgui.vert"), include_str!("imgui.frag"))
    }
}

/// A container for a bindable texture to be used internally.
pub struct Texture {
    bind_group: BindGroup,
}

impl Texture {
    /// Creates a new imgui texture from a wgpu texture.
    pub fn new(texture: wgpu::Texture, layout: &BindGroupLayout, device: &Device) -> Self {
        // Extract the texture view.
        let view = texture.create_default_view();

        // Create the texture sampler.
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("ImGui Sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });

        // Create the texture bind group from the layout.
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout,
            bindings: &[
                Binding {
                    binding: 0,
                    resource: BindingResource::TextureView(&view),
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
        });

        Texture { bind_group }
    }
}

pub struct Renderer {
    pipeline: RenderPipeline,
    uniform_buffer: Buffer,
    uniform_bind_group: BindGroup,
    textures: Textures<Texture>,
    texture_layout: BindGroupLayout,
    clear_color: Option<Color>,

    vertex_buffer_capacity: u64,
    index_buffer_capacity: u64,
    index_buffer: Buffer,
    vertex_buffer: Buffer,
}

impl Renderer {
    /// Create a new imgui wgpu renderer with newly compiled shaders.
    #[cfg(feature = "glsl-to-spirv")]
    pub fn new_glsl(
        imgui: &mut Context,
        device: &Device,
        queue: &mut Queue,
        format: TextureFormat,
        clear_color: Option<Color>,
    ) -> Renderer {
        let (vs_code, fs_code) = Shaders::get_program_code();
        let vs_raw = Shaders::compile_glsl(vs_code, ShaderStage::Vertex);
        let fs_raw = Shaders::compile_glsl(fs_code, ShaderStage::Fragment);
        Self::new_impl(imgui, device, queue, format, clear_color, vs_raw, fs_raw)
    }

    /// Create a new imgui wgpu renderer, using prebuilt spirv shaders.
    pub fn new(
        imgui: &mut Context,
        device: &Device,
        queue: &mut Queue,
        format: TextureFormat,
        clear_color: Option<Color>,
    ) -> Renderer {
        let vs_bytes = include_bytes!("imgui.vert.spv");
        let fs_bytes = include_bytes!("imgui.frag.spv");

        fn compile(shader: &[u8]) -> Vec<u32> {
            let mut words = vec![];
            for bytes4 in shader.chunks(4) {
                words.push(u32::from_le_bytes([
                    bytes4[0], bytes4[1], bytes4[2], bytes4[3],
                ]));
            }
            words
        }

        Self::new_impl(
            imgui,
            device,
            queue,
            format,
            clear_color,
            compile(vs_bytes),
            compile(fs_bytes),
        )
    }

    #[deprecated(note = "Renderer::new now uses static shaders by default")]
    pub fn new_static(
        imgui: &mut Context,
        device: &Device,
        queue: &mut Queue,
        format: TextureFormat,
        clear_color: Option<Color>,
    ) -> Renderer {
        Renderer::new(imgui, device, queue, format, clear_color)
    }

    /// Create an entirely new imgui wgpu renderer.
    fn new_impl(
        imgui: &mut Context,
        device: &Device,
        queue: &mut Queue,
        format: TextureFormat,
        clear_color: Option<Color>,
        vs_raw: Vec<u32>,
        fs_raw: Vec<u32>,
    ) -> Renderer {
        // Load shaders.
        let vs_module = device.create_shader_module(wgpu::ShaderModuleSource::SpirV(&vs_raw));
        let fs_module = device.create_shader_module(wgpu::ShaderModuleSource::SpirV(&fs_raw));

        // Create the uniform matrix buffer.
        let uniform_buffer_size = 64;
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: uniform_buffer_size,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });

        // Create the uniform matrix buffer bind group layout.
        let uniform_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            bindings: &[BindGroupLayoutEntry::new(
                0,
                wgpu::ShaderStage::VERTEX,
                BindingType::UniformBuffer {
                    dynamic: false,
                    min_binding_size: wgpu::NonZeroBufferAddress::new(uniform_buffer_size),
                },
            )],
        });

        // Create the uniform matrix buffer bind group.
        let uniform_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &uniform_layout,
            bindings: &[Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(uniform_buffer.slice(..)),
            }],
        });

        // Create the texture layout for further usage.
        let texture_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            bindings: &[
                BindGroupLayoutEntry::new(
                    0,
                    wgpu::ShaderStage::FRAGMENT,
                    BindingType::SampledTexture {
                        multisampled: false,
                        component_type: TextureComponentType::Float,
                        dimension: TextureViewDimension::D2,
                    },
                ),
                BindGroupLayoutEntry::new(
                    1,
                    wgpu::ShaderStage::FRAGMENT,
                    BindingType::Sampler { comparison: false },
                ),
            ],
        });

        // Create the render pipeline layout.
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            bind_group_layouts: &[&uniform_layout, &texture_layout],
        });

        // Create the render pipeline.
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            layout: &pipeline_layout,
            vertex_stage: ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(RasterizationStateDescriptor {
                front_face: FrontFace::Cw,
                cull_mode: CullMode::None,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: PrimitiveTopology::TriangleList,
            color_states: &[ColorStateDescriptor {
                format,
                color_blend: BlendDescriptor {
                    src_factor: BlendFactor::SrcAlpha,
                    dst_factor: BlendFactor::OneMinusSrcAlpha,
                    operation: BlendOperation::Add,
                },
                alpha_blend: BlendDescriptor {
                    src_factor: BlendFactor::OneMinusDstAlpha,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
                write_mask: ColorWrite::ALL,
            }],
            depth_stencil_state: None,
            vertex_state: VertexStateDescriptor {
                index_format: IndexFormat::Uint16,
                vertex_buffers: &[VertexBufferDescriptor {
                    stride: size_of::<DrawVert>() as BufferAddress,
                    step_mode: InputStepMode::Vertex,
                    attributes: &[
                        VertexAttributeDescriptor {
                            format: VertexFormat::Float2,
                            shader_location: 0,
                            offset: 0,
                        },
                        VertexAttributeDescriptor {
                            format: VertexFormat::Float2,
                            shader_location: 1,
                            offset: 8,
                        },
                        VertexAttributeDescriptor {
                            format: VertexFormat::Uint,
                            shader_location: 2,
                            offset: 16,
                        },
                    ],
                }],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        const INITIAL_VERTEX_BUFFER_CAPACITY: u64 = 2048;
        const INITIAL_INDEX_BUFFER_CAPACITY: u64 = INITIAL_VERTEX_BUFFER_CAPACITY * 4;

        let mut renderer = Renderer {
            pipeline,
            uniform_buffer,
            uniform_bind_group,
            textures: Textures::new(),
            texture_layout,
            clear_color,

            vertex_buffer_capacity: INITIAL_VERTEX_BUFFER_CAPACITY,
            index_buffer_capacity: INITIAL_INDEX_BUFFER_CAPACITY,
            vertex_buffer: Self::create_vertex_buffer(device, INITIAL_VERTEX_BUFFER_CAPACITY),
            index_buffer: Self::create_index_buffer(device, INITIAL_INDEX_BUFFER_CAPACITY),
        };

        // Immediately load the fon texture to the GPU.
        renderer.reload_font_texture(imgui, device, queue);

        renderer
    }

    fn create_vertex_buffer(device: &Device, num_vertices: u64) -> Buffer {
        device.create_buffer(&BufferDescriptor {
            label: Some("ImGui Vertex Buffer"),
            size: num_vertices * size_of::<DrawVert>() as u64,
            usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn create_index_buffer(device: &Device, num_indices: u64) -> Buffer {
        device.create_buffer(&BufferDescriptor {
            label: Some("ImGui Index Buffer"),
            size: num_indices * size_of::<DrawIdx>() as u64,
            usage: BufferUsage::INDEX | BufferUsage::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Render the current imgui frame.
    pub fn render<'r>(
        &'r mut self,
        draw_data: &DrawData,
        device: &Device,
        encoder: &'r mut CommandEncoder,
        queue: &Queue,
        view: &TextureView,
    ) -> RendererResult<()> {
        let fb_width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
        let fb_height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];

        // If the render area is <= 0, exit here and now.
        if !(fb_width > 0.0 && fb_height > 0.0) {
            return Ok(());
        }

        let width = draw_data.display_size[0];
        let height = draw_data.display_size[1];

        // Create and update the transform matrix for the current frame.
        // This is required to adapt to vulkan coordinates.
        // let matrix = [
        //     [2.0 / width, 0.0, 0.0, 0.0],
        //     [0.0, 2.0 / height as f32, 0.0, 0.0],
        //     [0.0, 0.0, -1.0, 0.0],
        //     [-1.0, -1.0, 0.0, 1.0],
        // ];
        let matrix = [
            [2.0 / width, 0.0, 0.0, 0.0],
            [0.0, 2.0 / -height as f32, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0, 1.0],
        ];
        self.update_uniform_buffer(queue, &matrix);

        // Start a new renderpass and prepare it properly.
        let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
            color_attachments: &[RenderPassColorAttachmentDescriptor {
                attachment: &view,
                resolve_target: None,
                load_op: match self.clear_color {
                    Some(_) => LoadOp::Clear,
                    _ => LoadOp::Load,
                },
                store_op: StoreOp::Store,
                clear_color: self.clear_color.unwrap_or(Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                }),
            }],
            depth_stencil_attachment: None,
        });
        rpass.push_debug_group("imgui");
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.uniform_bind_group, &[]);

        // Resize vertex & index buffer if necessary.
        {
            let required_vertex_buffer_size = draw_data
                .draw_lists()
                .map(|draw_list| draw_list.vtx_buffer().len())
                .sum::<usize>() as u64;
            if required_vertex_buffer_size > self.vertex_buffer_capacity {
                self.vertex_buffer_capacity = required_vertex_buffer_size;
                self.vertex_buffer =
                    Self::create_vertex_buffer(device, self.vertex_buffer_capacity);
            }
        }
        {
            let required_index_buffer_size = draw_data
                .draw_lists()
                .map(|draw_list| (draw_list.idx_buffer().len() + 3) / 4 * 4)
                .sum::<usize>() as u64;
            if required_index_buffer_size > self.index_buffer_capacity {
                self.index_buffer_capacity = required_index_buffer_size;
                self.index_buffer = Self::create_index_buffer(device, self.index_buffer_capacity);
            }
        }

        // Transfer data and execute all the imgui render work.
        let mut vertex_offset = 0;
        let mut index_offset = 0;
        for draw_list in draw_data.draw_lists() {
            queue.write_buffer(
                &self.vertex_buffer,
                vertex_offset * size_of::<DrawVert>() as BufferAddress,
                as_byte_slice(draw_list.vtx_buffer()),
            );

            // Hack to ensure that index buffer data length is a multiple of four (webgpu requirement).
            // We "know" this won't crash since arrays are 4 byte aligned
            let index_data = unsafe {
                let index_data = as_byte_slice(draw_list.idx_buffer());
                std::slice::from_raw_parts(index_data.as_ptr(), (index_data.len() + 3) / 4 * 4)
            };

            queue.write_buffer(
                &self.index_buffer,
                index_offset * size_of::<DrawIdx>() as BufferAddress,
                index_data,
            );

            self.render_draw_list(
                &mut rpass,
                &draw_list,
                draw_data.display_pos,
                draw_data.framebuffer_scale,
                vertex_offset,
                index_offset,
            )?;

            vertex_offset += draw_list.vtx_buffer().len() as u64;
            index_offset += (draw_list.idx_buffer().len() as u64 + 3) / 4 * 4;
        }
        rpass.pop_debug_group();

        Ok(())
    }

    /// Render a given `DrawList` from imgui onto a wgpu frame.
    fn render_draw_list<'render>(
        &'render self,
        rpass: &mut RenderPass<'render>,
        draw_list: &DrawList,
        clip_off: [f32; 2],
        clip_scale: [f32; 2],
        vertex_offset: u64,
        index_offset: u64,
    ) -> RendererResult<()> {
        let mut start = 0;
        rpass.push_debug_group("imgui - draw list");

        rpass.set_index_buffer(self.index_buffer.slice(
            (index_offset * size_of::<DrawIdx>() as u64)
                ..((index_offset + draw_list.idx_buffer().len() as u64)
                    * size_of::<DrawIdx>() as u64),
        ));
        rpass.set_vertex_buffer(
            0,
            self.vertex_buffer.slice(
                (vertex_offset * size_of::<DrawVert>() as u64)
                    ..((vertex_offset + draw_list.vtx_buffer().len() as u64)
                        * size_of::<DrawVert>() as u64),
            ),
        );

        for cmd in draw_list.commands() {
            if let Elements { count, cmd_params } = cmd {
                let clip_rect = [
                    (cmd_params.clip_rect[0] - clip_off[0]) * clip_scale[0],
                    (cmd_params.clip_rect[1] - clip_off[1]) * clip_scale[1],
                    (cmd_params.clip_rect[2] - clip_off[0]) * clip_scale[0],
                    (cmd_params.clip_rect[3] - clip_off[1]) * clip_scale[1],
                ];

                // Set the current texture bind group on the renderpass.
                let tex = self
                    .textures
                    .get(cmd_params.texture_id)
                    .ok_or_else(|| RendererError::BadTexture(cmd_params.texture_id))?;
                rpass.set_bind_group(1, &tex.bind_group, &[]);

                // Set scissors on the renderpass.
                let scissors = (
                    clip_rect[0].max(0.0).floor() as u32,
                    clip_rect[1].max(0.0).floor() as u32,
                    (clip_rect[2] - clip_rect[0]).abs().ceil() as u32,
                    (clip_rect[3] - clip_rect[1]).abs().ceil() as u32,
                );
                rpass.set_scissor_rect(scissors.0, scissors.1, scissors.2, scissors.3);

                // Draw the current batch of vertices with the renderpass.
                let end = start + count as u32;
                rpass.draw_indexed(start..end, 0, 0..1);
                start = end;
            }
        }

        rpass.pop_debug_group();
        Ok(())
    }

    /// Updates the current uniform buffer containing the transform matrix.
    fn update_uniform_buffer(&mut self, queue: &Queue, matrix: &[[f32; 4]; 4]) {
        let data = as_byte_slice(matrix);
        queue.write_buffer(&self.uniform_buffer, 0, data);
    }

    /// Updates the texture on the GPU corresponding to the current imgui font atlas.
    ///
    /// This has to be called after loading a font.
    pub fn reload_font_texture(&mut self, imgui: &mut Context, device: &Device, queue: &mut Queue) {
        let mut atlas = imgui.fonts();
        let handle = atlas.build_rgba32_texture();
        let font_texture_id =
            self.upload_texture(device, queue, &handle.data, handle.width, handle.height);

        atlas.tex_id = font_texture_id;
    }

    /// Creates and uploads a new wgpu texture made from the imgui font atlas.
    pub fn upload_texture(
        &mut self,
        device: &Device,
        queue: &mut Queue,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> TextureId {
        // Create the wgpu texture.
        let texture = device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width,
                height,
                depth: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsage::SAMPLED | TextureUsage::COPY_DST,
        });

        queue.write_texture(
            TextureCopyView {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d { x: 0, y: 0, z: 0 },
            },
            data,
            wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: data.len() as u32 / height,
                rows_per_image: height,
            },
            wgpu::Extent3d {
                width,
                height,
                depth: 1,
            },
        );

        let texture = Texture::new(texture, &self.texture_layout, device);
        self.textures.insert(texture)
    }
}

fn as_byte_slice<T>(slice: &[T]) -> &[u8] {
    let len = slice.len() * std::mem::size_of::<T>();
    let ptr = slice.as_ptr() as *const u8;
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

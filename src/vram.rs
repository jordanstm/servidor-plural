use wgpu::{Instance, Adapter, Device, Queue, InstanceDescriptor, Backends, RequestAdapterOptions};
use anyhow::{Result, anyhow};
use wgpu::util::DeviceExt;

pub struct PhoenixVramMotor {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub compute_pipeline: wgpu::ComputePipeline,
}

impl PhoenixVramMotor {
    pub async fn init() -> Result<Self> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance.request_adapter(&RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or_else(|| anyhow!("Falha ao encontrar uma GPU compatível (wgpu)"))?;

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("PhoenixMyceliumDevice"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await?;

        // Carregar Shader de Similaridade
        let shader_src = include_str!("similarity.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SimilarityShader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SimilarityPipeline"),
            layout: None,
            module: &shader,
            entry_point: "main",
        });

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            compute_pipeline,
        })
    }

    pub fn get_vram_info(&self) -> String {
        let info = self.adapter.get_info();
        format!("Device: {} | Backend: {:?}", info.name, info.backend)
    }
}

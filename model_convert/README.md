# 模型转换

## 导出模型（ONNX）
导出edsr onnx可以参考：https://github.com/sanghyun-son/EDSR-PyTorch/blob/master/src/main.py

在main.py加上如下代码，可以正常导出onnx：
```
model = model.to('cpu')
target_onnx_file = './edsr_baseline_x2_1.onnx'
dummy_input = torch.randn(1, 3, 1080, 1920)
idx_scale = 0
torch.onnx.export(model,
				 (dummy_input, idx_scale),
				  target_onnx_file,
				  export_params=True,
				  opset_version=11,
				  do_constant_folding=True,
				  dynamic_axes = {},
				  )
											   
print(f"Export model onnx to {target_onnx_file} finished")
```
这里固定onnx输入尺寸为：1x3x1080x1920

## 动态onnx转静态
```
onnxsim edsr_baseline_x2_1.onnx  edsr_baseline_x2_1_sim.onnx --overwrite-input-shape=1,1,1080,1920
```

## 转换模型（ONNX -> Axera）
使用模型转换工具 `Pulsar2` 将 ONNX 模型转换成适用于 Axera 的 NPU 运行的模型文件格式 `.axmodel`，通常情况下需要经过以下两个步骤：

- 生成适用于该模型的 PTQ 量化校准数据集
- 使用 `Pulsar2 build` 命令集进行模型转换（PTQ 量化、编译），更详细的使用说明请参考 [AXera Pulsar2 工具链指导手册](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)

### 量化数据集
准备量化图片若张，打包成Image.zip

### 模型转换

#### 修改配置文件
 
检查`config.json` 中 `calibration_dataset` 字段，将该字段配置的路径改为上一步下载的量化数据集存放路径  

#### Pulsar2 build

参考命令如下：

```
pulsar2 build --input edsr_baseline_x2_1.onnx --config ./build_config_edsr.json --output_dir ./output --output_name edsr_baseline_x2_1.axmodel  --target_hardware AX650 --compiler.check 0

也可将参数写进json中，直接执行：
pulsar2 build --config ./build_config_edsr.json
```

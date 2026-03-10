import torch
import torch.nn as nn

# 定义文档中的 ToyModel
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x

# 1. 初始化模型并移至 GPU
model = ToyModel(in_features=16, out_features=2).cuda()
x = torch.randn(8, 16).cuda()
target = torch.randn(8, 2).cuda()

# ==========================================
# 如何获取不同部分的数据类型：
# ==========================================

# 1. 获取【模型参数】的数据类型
# 遍历 parameters() 或者直接访问某一层的 weight
print("1. 模型参数 (fc1.weight):", model.fc1.weight.dtype)

# 2 & 3. 获取【中间层输出】的数据类型
# 推荐使用 Forward Hook 截获中间输出，这样不需要修改原模型的 forward 代码
def check_output_dtype_hook(module_name):
    def hook(module, input, output):
        print(f"-> {module_name} 输出数据类型:", output.dtype)
    return hook

# 给 fc1 和 ln 注册 Hook
model.fc1.register_forward_hook(check_output_dtype_hook("fc1 (First Linear)"))
model.ln.register_forward_hook(check_output_dtype_hook("ln (LayerNorm)"))

# 开启 FP16 混合精度上下文
with torch.autocast(device_type="cuda", dtype=torch.float16):
    
    # 4. 获取【Logits (模型最终输出)】的数据类型
    logits = model(x)
    print("4. Logits 数据类型:", logits.dtype)
    
    # 5. 获取【Loss】的数据类型
    # 注意：通常我们需要将 logits 转回 float32 再算 loss，但 autocast 有时会自己处理
    loss = torch.nn.functional.mse_loss(logits, target)
    print("5. Loss 数据类型:", loss.dtype)

# 反向传播
loss.backward()

# 6. 获取【模型梯度】的数据类型
# 梯度存储在参数的 .grad 属性中
print("6. 梯度 (fc1.weight.grad):", model.fc1.weight.grad.dtype)
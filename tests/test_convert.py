import torch
from torch2trt_dynamic import TRTModule, module2trt
from torchvision.models import resnet18


def test_convert(tmp_path):
    model = resnet18().cuda().eval()

    trt_model = module2trt(
        model,
        args=[torch.rand(1, 3, 32, 32).cuda()],
    )

    model_path = tmp_path / 'tmp.pth'
    torch.save(trt_model.state_dict(), model_path)
    assert model_path.exists()

    trt_model = TRTModule()
    trt_model.load_state_dict(torch.load(model_path))

    x = torch.rand(1, 3, 32, 32).cuda()
    with torch.no_grad():
        y = model(x)
        y_trt = trt_model(x)

    print(y)
    print(y_trt)
    torch.testing.assert_close(y, y_trt)

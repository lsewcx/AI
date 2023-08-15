import paddle
from paddlespeech.cli.cls import CLSExecutor

cls_executor = CLSExecutor()
result = cls_executor(
    model='panns_cnn14',
    config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
    label_file=None,
    ckpt_path=None,
    audio_file='./chirp.wav',
    topk=10,
    device=paddle.get_device())
print('CLS Result: \n {}'.format(result))

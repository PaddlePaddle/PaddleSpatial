import paddle
from PIL import Image
from paddlenlp.transformers import AutoProcessor, CLIPVisionModel
from clip import tokenize, load_model

device = paddle.set_device("gpu:1" if paddle.is_compiled_with_cuda() else "cpu")


def process_image(processor, visual_model, ii):
    img_tensor = processor(ii).unsqueeze(0).to(device)
    image_features = visual_model.encode_image(image)
    return image_features.detach()



image_type_dict = paddle.load('./image_type_dict.pdparams')



visual_model, processor = load_model('ViT_B_32', pretrained=True)
visual_model = visual_model.to(device)
visual_model.eval()


image_trans = {}
image_trans_rotate = {}
num = 0
with paddle.no_grad():
    for k, v in image_type_dict.items():
        try:
            path = f'./images/default_{k}.png'
            image = Image.open(path)
            rotate_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image_trans[k] =  process_image(processor, visual_model, image)
            image_trans_rotate[k] = process_image(processor, visual_model, rotate_image)
            print(num)
            num += 1
        except Exception as e:
            print(f"Error processing {k}: {e}")
            continue

paddle.save(image_trans, './clip_features.pdparams')
paddle.save(image_trans_rotate, './clip_features_rotate.pdparams')
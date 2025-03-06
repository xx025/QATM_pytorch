from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import torchvision
from qatm_pytorch import (
    ImageDataset,
    CreateModel,
    nms,
    run_multi_sample,
    plot_result_multi,
    nms_multi,
    run_one_sample,
)
from torchvision.models import VGG19_Weights
from torchvision import models


model = CreateModel(
    model=models.vgg19(weights=VGG19_Weights.DEFAULT).features,
    alpha=25,
    use_cuda=True,
)

template_dir = "template/"
image_path = "sample/sample1.jpg"
dataset = ImageDataset(Path(template_dir), image_path, thresh_csv="thresh_template.csv")


def test_multi_sample():

    scores, w_array, h_array, thresh_list = run_multi_sample(model, dataset)

    boxes, indices = nms_multi(scores, w_array, h_array, thresh_list)

    d_img = plot_result_multi(
        dataset.image_raw, boxes, indices, show=True, save_name="result_sample.png"
    )

    plt.imshow(scores[2])


def test_one_sample():
    
    
    idx=0

    for data in dataset:

        w = data["template_w"]
        h = data["template_h"]

        score = run_one_sample(
            model, data["template"], data["image"], data["thresh"]
        )

        score=score[0] # only one sample
        nms_one = nms(score=score, w_ini=w, h_ini=h, thresh=0.7)
        print(nms_one)
        
        indices=np.full(shape=[len(nms_one)],dtype=int,fill_value=idx)
        
        d_img = plot_result_multi(
            dataset.image_raw, nms_one, indices, show=True, save_name=f"result_sample_{idx}.png"
        )
        
        idx+=1
        pass


if __name__ == "__main__":

    # multi sample
    # test_multi_sample( )

    # single sample
    test_one_sample()

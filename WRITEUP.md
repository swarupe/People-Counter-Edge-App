## Explaining Custom Layers

Cusomer layers are something which are not supported natively by the framework. Currently when I was converting the tensorflow pretrained model 
for inference I got some of the custom layers. Looking into documentation I got to know that I was supposed to use cpu extension plugin. So I followed it
and got my model to work.

Reading Documentation gave me 2 methods to convert custom layer
- One which I used which involves in adding extension plugin
- Second method varies based on framework. It involves registering the operation performed by the layer. Attributes that are supported by the custom layer and computing the output shape for each instance of the custom layer from its parameters

Reasons why we need to handle custom layers are..
- First reason is that to use model for inference I need to convert all custom layers.
- Second reason is that Model Optimizer does not know about the custom layers so it needs to taken care of and also need to handle for handle unsupported layers at the time of inference.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was very less. I could almost see both were having same results. But I could see 6% difference
in my script where openvino performed slightly better.

The size of the model pre- and post-conversion was 67MB and 65MB respectively. There was 2MB difference.

The inference time of the model pre- and post-conversion was very much noticable. Before convertion it was taking around 93.36ms on an average. But after converting I got 3.36ms. Openvino IR model are much faster compared to the model before IR.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...
- First use case which I can think of now is for covid-19 prevention. We can monitor crowd using this and can give alert to people.
- We can use it in malls or any commercials places to analyze in what time more crowd is present. Based on that we can send assistance to handle crowd.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

- Lighting: Model will not perform well in dark lighting. So having better source of light is must. 
- Model accuracy: Model must have better accuracy say 90+ or else we will not get better results. High number of wrong predictions will lead to bigger problems.
- Camera focal lenght: Focal length can vary based on use cases. Higher focal length required for wider area and lower focal length for narrow areas.
- Image size / Resolution: This will depend on many factors, model will always perform better if same resolution of images feeded which model has trained. If low resolution is given then there will be chance of giving bad predictions.

## Conclusion

I had [ssd_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz) from tensorflow object detection model zoo. Though performance is not that great but it is enough for running for our use case. I see [person detection models](https://github.com/opencv/open_model_zoo/blob/master/models/intel/index.md) from intel ovenvino model zoo. Which can give better results because those trained on people images. 

I downloaded the model using downloaded and ran directly with it, which gave accurate results. But for this submission I am uploading with ssd mobilenet v2 coco model.

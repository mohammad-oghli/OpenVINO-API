# OpenVINO API Daisi

This web service is an implementation of Intel **OpenVINO** API on Daisi Platform which provide optimized deep learning inference.

It provides OpenVINO Models as a service in order to present the capabilities of these models for ML applications.

At this time It contains 8 different deployed models which available on [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) for OpenVINO Toolkit.

This implementation in its initial phase and still need alot of work.

Check this [demo](https://youtu.be/-VdZlzWaJvA) for more info about the project.
____________________________________
## Models
Currently, Deployed OpenVINO Models:
* Animal Classification 

  ![animal_classify](https://i.imgur.com/yj7Epmy.png)
   
  Animal Class: `'cheetah, chetah, Acinonyx jubatus'`


* Road Segmentation 
  
  ![road_segmentation](https://i.imgur.com/uvTwSB1.png)

* Optical Character Detection(OCD) 
  
  ![ocd](https://i.imgur.com/2RINxrL.png)

* Super Resolution 

  ![superresulotion](https://i.imgur.com/cIC2Hx8.png)

* Vehicle Recognition 
 
  ![vehicle_rec](https://i.imgur.com/ApGsWxJ.png)

* Named Entity Recognition(NER) 
  
  <pre>source_text = "Intel was founded in Mountain View, California, " \
                  "in 1968 by Gordon E. Moore, a chemist, and Robert Noyce, " \
                  "a physicist and co-inventor of the integrated circuit."
  
  {
    'Extraction': [
        {'Entity': 'Intel', 'Type': 'company', 'Score': '0.98'},
        {
            'Entity': 'Gordon E. Moore, a chemist, and Robert Noyce',
            'Type': 'persons',
            'Score': '0.83'
        },
        {'Entity': 'Mountain View', 'Type': 'city', 'Score': '0.79'},
        {'Entity': 'California', 'Type': 'state', 'Score': '0.98'}
    ]
  }
</pre>

* Handwritten OCR (Chinese & Japanese) 
 
  ![handwritten_ocr](https://i.imgur.com/EcqdEP1.png)
  
  Recognized Text: `'人有悲欢离合，月有阴睛圆缺，此事古难全。'`


* Interactive Question Answering 
  
  <pre>
  sources = ["https://en.wikipedia.org/wiki/OpenVINO"]
  question = "What does OpenVINO refers to?"
  
  {
      'Question': 'What does OpenVINO refers to?',
      'Answer': 'Open Visual Inference and Neural network Optimization',
      'Score': '0.85'
  } </pre>


## How to call it

-First load **OpenVINO API** Daisi

<pre>
import pydaisi as pyd
openvino_api_v3 = pyd.Daisi("oghli/OpenVINO API v3")
</pre>

* **Animal Classification Model**

  Call the `cv_animal_classify` end point, passing the `image_source` to classify it, you can pass image source either from `data/animal_classify/` directory or from valid `url` of the image
  <pre>
  img = "data/animal_classify/test.jpg"
  result = openvino_api_v3.cv_animal_classify(img).value
  result
  </pre>

* **Road Segmentation Model**
    
  Call the `cv_road_segmentation` end point, passing the `image_source` to segment roads in it, you can pass image source either from `data/road_segmentation/` directory or from valid `url` of the image
  <pre>
  img = "data/road_segmentation/empty_road_mapillary.jpg"
  result = openvino_api_v3.cv_road_segmentation(img).value
  </pre>
  To show the result image import `matplotlib` then use `plt.imshow` method
  <pre>
  import matplotlib.pyplot as plt
  plt.figure(figsize=(15, 5))
  plt.imshow(result)
  </pre>

* **Optical Character Detection Model**

  Call the `cv_ocd` end point, passing the `image_source` to detect text in it, you can pass image source either from `data/ocr/` directory or from valid `url` of the image
  <pre>
  img = "data/ocr/intel_rnb.jpg"
  result = openvino_api_v3.cv_ocd(img).value
  plt.figure(figsize=(20, 10))
  plt.imshow(result)
  </pre>

* **Super Resolution Model** 

  Call the `cv_superresolution` end point, passing the `image_source` to enhance it, you can pass image source either from a directory path or from valid `url` of the image
  <pre>
  img = "https://i.imgur.com/R5ovXDO.jpg"
  result = openvino_api_v3.cv_superresolution(img).value
  plt.figure(figsize=(20, 10))
  plt.imshow(result[1])
  </pre>

* **Vehicle Recognition Model**
  
  Call the `cv_vehicle_rec` end point, passing the `image_source` to recognize Vehicles in it, you can pass image source either from a directory path or from valid `url` of the image
  
  <pre>
  img = "https://i.imgur.com/IvwQdz5.jpg"
  result = openvino_api_v3.cv_vehicle_rec(img).value
  plt.figure(figsize=(20, 10))
  plt.imshow(result)
  </pre>

* **Named Entity Recognition Model**
    
  Call the `analyze_entities` end point, passing the `source_text` to analyze entities in it
  <pre>
  source_text = "Intel was founded in Mountain View, California, " \
                    "in 1968 by Gordon E. Moore, a chemist, and Robert Noyce, " \
                    "a physicist and co-inventor of the integrated circuit."
  result = openvino_api_v3.analyze_entities(source_text).value
  result 
  </pre>

* **Handwritten OCR Model**
    
  Call the `handwritten_ocr` end point, passing the `image_source` and the selected language `lang` to recognize text in it, you can pass image source either from `data/handwritten_ocr/` directory or from valid `url` of the image
  <pre>
  img = "data/handwritten_ocr/handwritten_chinese_test.jpg"
  result = openvino_api_v3.handwritten_ocr(img, "ch").value
  result
  </pre>
  
  Set lang using `"ch"` for chinesse language or `"jap"` for japanese language
   
  <pre>img = "data/handwritten_ocr/handwritten_japanese_test.png"
  result = openvino_api_v3.handwritten_ocr(img, "jap").value
  result 
  </pre>

* **Interactive Question Answering Model**

    Call the `question_answering` end point, passing the `sources` as array and the `question` to answer it using the provided sources, you can pass sources either as string `text` or from valid `url` for the information resource
  
  <pre>
  sources = ["https://en.wikipedia.org/wiki/OpenVINO"]
  question = "What does OpenVINO refers to?"
  result = openvino_api_v3.question_answering(sources, question).value
  result
  </pre>
  
  <pre>
  sources = ["Computational complexity theory is a branch of the theory of computation in theoretical computer "
             "science that focuses on classifying computational problems according to their inherent difficulty, "
             "and relating those classes to each other. A computational problem is understood to be a task that "
             "is in principle amenable to being solved by a computer, which is equivalent to stating that the "
             "problem may be solved by mechanical application of mathematical steps, such as an algorithm."]
  
  question = "what branch of the theory of computation in theoretical computer science that focuses on classifying computational problems?"
  result = openvino_api_v3.question_answering(sources, example_question=question).value
  result
  </pre> 
____________________________________

Please support me by starring the repository.

Contributor: _**Mohamad Oghli**_.

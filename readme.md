# OpenVINO API Daisi

This web service is an implementation of Intel **OpenVINO** API on Daisi Platform which provide optimized deep learning inference.

It provides OpenVINO Models as a service in order to present the capabilities of these models for ML applications.

At this time It contains 8 different deployed models which available on [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) for OpenVINO Toolkit.

This implementation in its initial phase and still need alot of work.

Check this [demo](https://youtu.be/-VdZlzWaJvA) for more info about the project.
____________________________________
### Models
Currently, Deployed OpenVINO Models:
* Animal Classification

  ![animal_classify](https://i.imgur.com/yj7Epmy.png)
   
  `'cheetah, chetah, Acinonyx jubatus'`


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
  
  `'人有悲欢离合，月有阴睛圆缺，此事古难全。'`


* Interactive Question Answering
  
  <pre>
  sources = ["https://en.wikipedia.org/wiki/OpenVINO"]
  question = "What does OpenVINO refers to?"
  
  {
      'Question': 'What does OpenVINO refers to?',
      'Answer': 'Open Visual Inference and Neural network Optimization',
      'Score': '0.85'
  } </pre>



____________________________________

Please support me by forking and starring the repository.

Contributor: _**Mohamad Oghli**_.

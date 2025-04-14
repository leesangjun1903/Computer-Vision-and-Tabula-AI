# Introduction
This book is an introduction to the broad field of computer vision. Without a doubt, machines can be built to see; for example, machines inspect millions of light bulb filaments and miles of fabric each day. Automatic teller machines (ATMs) have been built to scan the human eye for user identification and cars have been driven by a computer using camera input. This chapter introduces several important problem areas where computer vision pro- vides solutions. After reading this chapter, you should have a broad view of some problems and methods of computer vision.

이 책은 컴퓨터 비전의 광범위한 분야를 소개하는 책입니다.  
의심할 여지 없이 기계는 매일 수백만 개의 전구 필라멘트와 수 마일의 천을 검사하는 등 볼 수 있도록 제작될 수 있습니다.  
자동 입출금기(ATM)는 사용자 식별을 위해 사람의 눈을 스캔하도록 제작되었으며, 자동차는 카메라 입력을 사용하여 컴퓨터로 구동되었습니다.  
이 장에서는 컴퓨터 비전이 솔루션을 제공하는 몇 가지 중요한 문제 영역을 소개합니다.  
이 장을 읽고 나면 컴퓨터 비전의 몇 가지 문제와 방법에 대해 폭넓게 살펴볼 수 있습니다.

n this book, we generally use the terms machine vision and computer vision to mean the same thing However, we often use the term machine vision in the context of industrial applications and the term computer vision with the field in general.

이 책에서는 일반적으로 머신 비전과 컴퓨터 비전이라는 용어를 같은 의미로 사용합니다.  
그러나 산업 응용 분야에서는 머신 비전이라는 용어를, 일반적으로는 컴퓨터 비전이라는 용어를 사용하는 경우가 많습니다.

##### 1 DEFINITION 
The goal of computer vision is to make useful decisions about real physical objects and scenes based on sensed images.

컴퓨터 비전의 목표는 감지된 이미지를 바탕으로 실제 물리적 객체와 장면에 대해 유용한 결정을 내리는 것입니다.

In order to make decisions about real objects, it is almost always necessary to construct some description or model of them from the image. Because of this, many experts will say that the goal of computer vision is the construction of scene descriptions from images. Although our study of computer vision is problem-oriented, fundamental issues will be ad- dressed. Critical issues raised in this chapter and studied in the remainder of the text include the following.

실제 물체에 대한 의사 결정을 내리기 위해서는 거의 항상 이미지에서 일부 설명이나 설명에 대한 모델을 구성해야 합니다.  
이 때문에 많은 전문가들은 컴퓨터 비전의 목표가 이미지에서 장면 설명을 구성하는 것이라고 말할 것입니다.  
컴퓨터 비전에 대한 우리의 연구는 문제 중심이지만 근본적인 문제는 다룰 것입니다. 이 장에서 제기하고 나머지 본문에서 연구한 중요한 문제는 다음과 같습니다.  

Sensing: How do sensors obtain images of the world? How do the images encode properties of the world, such as material, shape, illumination and spatial relationships? Encoded Information: How do images yield information for understanding the 3D world, including the geometry, texture, motion, and identity of objects in it? Representations: What representations should be used for stored descriptions of objects, their parts, properties and relationships? Algorithms: What methods are there to process image information and construct descrip- tions of the world and its objects?

감지: 센서는 어떻게 세상의 이미지를 얻나요? 이미지는 재료, 모양, 조명 및 공간 관계와 같은 세계의 속성을 어떻게 인코딩하나요?  
인코딩된 정보: 이미지는 어떻게 3D 세계를 이해하는 데 필요한 기하학적 구조, 질감, 움직임, 사물의 정체성 등의 정보를 제공하나요?  
표현: 객체, 그 부분, 속성 및 관계에 대한 저장된 설명에는 어떤 표현을 사용해야합니까?  
알고리즘: 이미지 정보를 처리하고 세계와 그 객체에 대한 설명을 구성하는 방법에는 어떤 것이 있습니까?  

These issues and others will be studied in the following chapters. We now introduce various applications and some important issues that arise in their context.

이러한 문제와 기타 문제는 다음 장에서 연구할 것입니다. 이제 다양한 응용 프로그램과 그 맥락에서 발생하는 몇 가지 중요한 문제를 소개합니다.

## 1.1 Machines that see?
Scientists and science fiction writers have been fascinated by the possibility of building in- telligent machines, and the capability of understanding the visual world is a prerequisite that some would require of such a machine. Much of the human brain is dedicated to vision. Humans solve many visual problems effortlessly, yet most have little analytical un- derstanding of visual cognition as a process. Allan Turing, one of the fathers of both the modern digital computer and the field of artificial intelligence, believed that a digital com- puter would achieve intelligence and the ability to understand scenes. Such lofty goals have proved difficult to achieve and the richness of human imagination is not yet matched by our engineering. However, there has been surprising progress along some lines of research. While building practical systems is a primary theme of this text and artificial intelligence is not, we will sometimes ponder the deeper questions, and, where we can, make some assessment of progress. Consider, for example, the following scenario, which could be realized within the next few years. A TV camera at your door provides images to your home computer which you have trained to recognize some faces of people important to you. When you call in to your home message center, your computer not only reports the phone messages, but it also reports probable visits from your sister Eleanor and Chad the paper boy. We will discuss such current research ideas at various places in the book.

과학자와 공상 과학 소설 작가들은 지능형 기계를 구축할 수 있는 가능성에 매료되어 왔으며, 시각 세계를 이해하는 능력은 일부 사람들이 이러한 기계에 필요로 하는 전제 조건입니다.  
인간의 뇌는 대부분 시각에 전념하고 있습니다. 인간은 많은 시각 문제를 쉽게 해결하지만, 대부분 시각 인지를 그 과정으로 분석적으로 이해하는 능력은 거의 없습니다.  
현대 디지털 컴퓨터와 인공지능 분야의 아버지 중 한 명인 앨런 튜링은 디지털 컴퓨터 사용자가 지능과 장면을 이해하는 능력을 달성할 수 있다고 믿었습니다.  
이러한 높은 목표는 달성하기 어려웠고 인간의 상상력의 풍부함은 아직 우리의 엔지니어링과 일치하지 않습니다.  
그러나 일부 연구 분야에서는 놀라운 진전이 있었습니다. 실용적인 시스템 구축은 이 텍스트와 인공지능의 주요 주제는 아니지만, 때로는 더 깊은 질문에 대해 고민하고 진행 상황을 평가할 수 있는 경우도 있습니다.  
예를 들어, 향후 몇 년 내에 실현될 수 있는 다음 시나리오를 생각해 보세요.  
집 앞에 있는 TV 카메라는 자신에게 중요한 사람들의 얼굴을 인식하도록 훈련한 이미지를 집 컴퓨터에 제공합니다.  
집 메시지 센터에 전화하면 컴퓨터가 전화 메시지를 보고할 뿐만 아니라 여동생 엘리너와 배달 소년 채드의 방문 가능성도 보고합니다.  
책의 여러 곳에서 이러한 최신 연구 아이디어에 대해 논의할 예정입니다.

## 1.2 Application problems
The applications of computers in image analysis are virtually limitless. Only a small sample of applications can be included here, but these will serve us well for both motivation and orientation to the field of study.

이미지 분석에서 컴퓨터의 응용은 사실상 무한합니다. 여기에는 소수의 응용 프로그램만 포함될 수 있지만, 이러한 응용 프로그램은 학습 분야에 대한 동기 부여와 방향성 모두에 도움이 될 것입니다.

### A preview of the digital image
A preview of the digital imageA digital image might represent a cartoon, a page of text, a person's face, a map of Kat- mandu, or a product for purchase from a catalog. A digital image contains a fixed number of rows and columns of pixels, short for picture elements. Pixels are like little tiles holding quantized values - small numbers, often between 0 and 255, that represent the brightness at the points of the image. Depending on the coding scheme, 0 could be the darkest and 255 the brightest, or visa-versa. At the top left in Figure 1.1 is a printed digital image of a face that is 257 rows high by 172 columns wide. At the top center is an 8 x 8 subimage ex- tracted from the right eye of the left image. At the bottom of the figure are the 64 numbers representing the brightness of the pixels in that subimage. The numbers below 100 in the upper right of the subimage represent the lower reflection from the dark of the eye iris), while the higher numbers represent the brighter white of the eye. A color image would have three numbers for each pixel, perhaps one value for red, one for blue, and one for green. Digital images are most commonly displayed on a monitor, which is basically a television screen with a digital image memory. A color image that has 500 rows and 500 columns is roughly equivalent to what you see at one instant of time on your TV. A pixel is displayed by energizing a small spot of luminescent material; displaying color requires energizing 3 neighboring spots of different materials. A high resolution computer display has roughly 1200 by 1000 pixels. The next chapter discusses digital images in more detail, while coding and interpretation of color in digital images is treated in Chapter 6.

디지털 이미지는 만화, 텍스트 페이지, 사람의 얼굴, 카트만두 지도 또는 카탈로그에서 구매할 수 있는 제품을 나타낼 수 있습니다.  
디지털 이미지에는 그림 요소의 줄임말로 고정된 수의 행과 열이 있는 픽셀이 포함되어 있습니다.  
픽셀은 양자화된 값을 가진 작은 타일과 같습니다. 작은 숫자(종종 0에서 255 사이)로 이미지 지점의 밝기를 나타냅니다.  
코딩 방식에 따라 0이 가장 어둡고 255가 가장 밝을 수 있으며, 아닌 경우도 있습니다.  

<img width="833" alt="스크린샷 2025-04-14 오후 1 24 18" src="https://github.com/user-attachments/assets/ab80ea69-eff7-4f92-a768-0c0daef28160" />

그림 1.1의 왼쪽 상단에는 가로 172열로 높이 257줄의 얼굴이 인쇄된 디지털 이미지입니다. 상단 중앙에는 왼쪽 이미지의 오른쪽 눈에서 추출한 8 x 8개의 하위 이미지가 있습니다.  
그림 하단에는 해당 하위 이미지의 픽셀 밝기를 나타내는 64개의 숫자가 있습니다. 하위 이미지 오른쪽 상단에 있는 100 미만의 숫자는 홍채의 어두운 부분에서 반사되는 낮은 숫자를 나타내며, 높은 숫자는 눈의 밝은 흰색을 나타냅니다.  
컬러 이미지는 각 픽셀에 대해 빨간색, 파란색, 녹색의 값으로 세 개의 숫자를 가질 수 있습니다.  
디지털 이미지는 가장 일반적으로 모니터에 표시되며, 이는 기본적으로 디지털 이미지 메모리가 있는 텔레비전 화면입니다.  
500줄과 500열로 구성된 컬러 이미지는 TV에서 한 순간에 보는 것과 대략적으로 동일합니다.  
픽셀은 발광 물질의 작은 부분에 에너지를 공급하여 표시되며, 색상을 표시하려면 서로 다른 물질의 인접한 3개의 부분에 에너지를 공급해야 합니다.  
고해상도 컴퓨터 디스플레이에는 약 1200 x 1000픽셀이 있습니다.  
다음 장에서는 디지털 이미지에 대해 자세히 설명하고, 디지털 이미지에서 색상의 코딩 및 해석은 6장에서 다룹니다.

### Image Database Query




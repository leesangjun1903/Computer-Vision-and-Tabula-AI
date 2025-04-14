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
Huge digital memories, high bandwidth transmission and multimedia personal computers have facilitated the development of image databases. Good use of the many existing images requires good retrieval methods. Standard database techniques apply to images that have been augmented with text keys; however, content-based retrieval is needed and is a topic of much current research. Suppose that a newly formed company wants to design and protect a new logo and that an artist has created several candidates for the company to consider. A logo cannot be used if it is too similar to one of an existing company, so a database of existing logos must be searched. This operation is analagous to patent search and is done by humans, but could be greatly aided by machine vision methods. See Figure 1.2. There are many similar problems. Suppose an architect or an art historian wants to search for buildings with a particular kind of entryway. It would be desirable to just provide a picture, perhaps fetched from the database itself, and request the system to produce other similar pictures. In a later chapter, you will see how geometric, color, and texture features can be used to aid in answering such an image database query. Suppose that an advertising agency wants to search for existing images of young children enjoying eating. This seman- tic requirement, which is simple for humans to understand, presents a very high level of difficulty for machine vision. Characterizing "children", "enjoyment", and "eating" would require complex use of color, texture, and geometric features. We note in passing that a computer algorithm has been devised that decides whether or not a color image contains a naked person. This could be useful for parents who want to screen images that their children retrieve from the web. Image database retrieval methods are treated in Chapter 8.

방대한 디지털 메모리, 고대역폭 전송 및 멀티미디어 개인용 컴퓨터는 이미지 데이터베이스의 개발을 촉진시켰습니다.  
기존의 많은 이미지를 잘 활용하려면 좋은 검색 방법이 필요합니다.  
표준 데이터베이스 기술은 텍스트 키로 보강된 이미지에 적용되지만, 콘텐츠 기반 검색이 필요하며 현재 많은 연구 주제입니다.  
새로 설립된 회사가 새로운 로고를 디자인하고 보호하고자 하고 한 아티스트가 회사가 고려할 후보를 여러 개 만들었다고 가정해 보겠습니다.  
로고가 기존 회사와 너무 유사하면 사용할 수 없으므로 기존 로고의 데이터베이스를 검색해야 합니다.  
이 작업은 특허 검색과 유사하며 인간이 수행하지만 머신 비전 방법의 도움을 크게 받을 수 있습니다.  
그림 1.2를 참조하세요. 유사한 문제가 많이 있습니다.  

<img width="833" alt="스크린샷 2025-04-14 오후 4 04 08" src="https://github.com/user-attachments/assets/08430145-fd2b-482b-b74f-96f8dc757bfb" />

건축가나 미술사학자가 특정 종류의 진입로가 있는 건물을 검색하고 싶다고 가정해 보겠습니다.  
데이터베이스 자체에서 가져온 그림을 제공하고 시스템에 다른 유사한 그림을 요청하면 바람직할 것입니다.  
이후 장에서는 이러한 이미지 데이터베이스 쿼리에 응답하는 데 기하학적, 색상 및 질감 특징을 어떻게 사용할 수 있는지 살펴볼 것입니다.  

광고 대행사가 먹는 것을 즐기는 어린 아이들의 기존 이미지를 검색하려고 한다고 가정해 보겠습니다.  
사람이 이해하기 쉬운 이 의미론적 요구 사항은 머신 비전에 매우 높은 수준의 난이도를 제공합니다.  
'어린이', '즐거움', '먹는 것'을 특성화하려면 색상, 질감, 기하학적 특징을 복합적으로 사용해야 합니다.  
색상 이미지에 벌거벗은 사람이 포함되어 있는지 여부를 결정하는 컴퓨터 알고리즘이 고안되었음을 참고합니다.  
이는 자녀가 웹에서 검색한 이미지를 스크리닝하려는 부모에게 유용할 수 있습니다. 이미지 데이터베이스 검색 방법은 8장에서 다룹니다.

### Inspecting crossbars for holes
In the late 1970's an engineer in Milwaukee implemented a machine vision system that successfully counted the number of bolt holes in crossbars made for truck companies. The truck companies demanded that every crossbar be inspected before being shipped to them, because a missing bolt hole on a partly assembled truck was a very costly defect. Either the assembly line would have to be stopped while the needed hole was drilled, or worse, a worker might ignore placing a required bolt in order to keep the production line running. To create a digital image of the truck crossbar, lights were placed beneath the existing transfer line and a digital camera above it. When a crossbar came into the field of view, an image was taken. Dark pixels inside the shadow of the crossbar were represented as 1's indicating steel, and pixels in the bright holes were represented as O's, indicating that the hole was drilled. The number of holes can be computed as the number of external corners minus the number of internal corners all divided by four. Figure 1.3 shows three bright holes ('O's) in a background of '1's. An external corner is just a 2 x 2 set of neighboring pixels containing exactly 3 ones while an internal corner is a 2 x 2 set of neighboring pixels containing exactly 3 zeroes. Example processing of an image with 7 rows and 33 columns is shown in the figure and a skeleton algorithm is also shown. Holecounting is only one example of many simple, but powerful operations possible with digital images. (As the exercises below show, the holecounting algorithm is correct only if the holes are "4-connected" and "simply connected" - that is, they have no background pixels inside them. These concepts are discussed further in Chapter 3 and in more detail in the text by Rosenfeld.)

1970년대 후반 밀워키의 한 엔지니어가 트럭 회사를 위해 만든 크로스바의 볼트 구멍 수를 성공적으로 세는 기계 비전 시스템을 구현했습니다.  
트럭 회사들은 부분 조립된 트럭의 볼트 구멍이 누락된 것은 매우 비용이 많이 드는 결함이었기 때문에 모든 크로스바를 배송하기 전에 검사할 것을 요구했습니다.  
필요한 구멍을 뚫는 동안 조립 라인을 멈춰야 하거나, 더 나쁜 경우 작업자가 생산 라인을 계속 가동하기 위해 필요한 볼트를 설치하는 것을 무시할 수도 있습니다.  
트럭 크로스바의 디지털 이미지를 만들기 위해 기존 이송 라인 아래에 조명을, 그 위에 디지털 카메라를 배치했습니다.  
크로스바가 시야에 들어오면 이미지를 촬영했습니다. 크로스바 그림자 내부의 어두운 픽셀은 강철을 나타내는 1로 표시되었고, 밝은 구멍의 픽셀은 구멍이 뚫렸다는 것을 나타내는 O로 표시되었습니다.  
구멍의 수는 외부 모서리의 수에서 내부 모서리의 수를 뺀 값으로 계산할 수 있습니다. 그림 1.3은 '1'의 배경에 있는 세 개의 밝은 구멍('O')을 보여줍니다.  

<img width="592" alt="스크린샷 2025-04-14 오후 4 07 59" src="https://github.com/user-attachments/assets/e89afde4-8804-468c-b928-d9c9806741b7" />

외부 모서리는 정확히 3개의 구멍을 포함하는 인접 픽셀의 2 x 2 세트이고, 내부 모서리는 정확히 3개의 0을 포함하는 인접 픽셀의 2 x 2 세트입니다.  
그림에는 7행 33열로 구성된 이미지의 예시 처리가 나와 있으며 골격 알고리즘도 표시되어 있습니다.  
홀 카운팅은 디지털 이미지로 가능한 간단하지만 강력한 작업의 한 예에 불과합니다. (아래 연습 문제에서 볼 수 있듯이 홀 카운팅 알고리즘은 구멍이 "4" 및 "단순히 연결"된 경우에만 정확하며, 즉 구멍 내부에 배경 픽셀이 없는 경우에만 정확합니다. 이러한 개념은 3장에서 자세히 설명하고 Rosenfeld의 본문에서 자세히 설명합니다.)

### Examining the inside of a human head.
Magnetic resonance imaging (MRI) devices can sense materials in the interior of 3D objects. Figure 1.4 shows a section through a human head: brightness is related to movement of material, so this is actually a picture of blood flow. One can "see" important blood vessels. 

자기공명영상(MRI) 장치는 3D 물체 내부의 물질을 감지할 수 있습니다.  
그림 1.4는 사람의 머리를 통과하는 단면을 보여줍니다: 밝기는 물질의 움직임과 관련이 있으므로 실제로는 혈류를 보여주는 그림입니다. 중요한 혈관을 "볼 수" 있습니다.

The whispy comet-like structures are associated with the eyes. MRI images are used by doctors to check for tumors or blood flow problems such as abnormal vessel constrictions or expansions. The image at the right in Figure 1.4 was made from a copy of the one on the left by making every pixel of value 208 or more bright (255) and those below 208 dark (0). Most pixels correctly show blood vessels versus background, but there are many incorrectly "colored" pixels of both types. Machine vision techniques are often used in medical image nalysis, although usually to aid in data presentation and measurement rather than diagnosis itself. Wouldn't it be great if we could "see" thoughts occuring in the brain! Well, it turns out that MRI can sense organic activity related to thought processes and this is a very exciting current area of research.

위스피 혜성과 같은 구조는 눈과 관련이 있습니다. MRI 영상은 의사가 종양이나 비정상적인 혈관 수축 또는 확장과 같은 혈류 문제를 확인하는 데 사용됩니다.  
그림 1.4의 오른쪽 이미지는 값 208 이상의 픽셀(255)과 208 미만의 픽셀(0)을 모두 밝게 만들어 왼쪽 이미지를 복사한 것입니다.  
대부분의 픽셀은 혈관과 배경을 정확하게 보여주지만 두 유형 모두 잘못 '색깔'이 있는 픽셀이 많습니다.  
의료 영상 분석에는 머신 비전 기술이 자주 사용되지만, 일반적으로 진단 자체보다는 데이터 표시 및 측정을 돕기 위해 사용됩니다.  
뇌에서 일어나는 생각을 '볼 수' 있다면 멋지지 않을까요? 음, MRI는 사고 과정과 관련된 유기적 활동을 감지할 수 있으며 현재 매우 흥미로운 연구 분야입니다.

### Processing scanned text pages
A common problem is to convert information from paper documents into digital form for information systems. For example, we might want to make an old book available on the Internet, or we might need to convert a blueprint of some object into a geometry file so that the part can be made by a numerically controlled machine tool. Figure 1.5 shows the same message in both Chinese and English. The Chinese charac- ters were written on paper and scanned into an image of 482 rows and 405 columns. The postscript file encoding the graphics and printed in the figure has a size of 68,464 bytes. The English version is stored in a file of 115 bytes, each holding one ASCII character. There is an entire range of important applications in processing documents. Recognizing individual characters from the dots of the scanner or FAX files is one such application that is done fairly well today, provided that the characters conform to standard patterns. Providing a semantic interpretation of the information, possibly to be used for indexing in a large database, is a harder problem.

일반적인 문제는 종이 문서의 정보를 정보 시스템용 디지털 형태로 변환하는 것입니다.  
예를 들어, 인터넷에서 오래된 책을 사용할 수 있도록 하거나, 어떤 물체의 청사진을 기하학 파일로 변환하여 숫자로 제어되는 공작 기계로 만들 수 있도록 해야 할 수도 있습니다.  
그림 1.5는 중국어와 영어 모두에서 동일한 메시지를 보여줍니다.  
한자는 종이에 쓰여 482행 405열의 이미지로 스캔되었습니다.  
그래픽을 인코딩하고 그림에 인쇄된 포스트스크립트 파일의 크기는 68,464바이트입니다.  
영어 버전은 115바이트의 파일에 저장되며, 각 파일에는 하나의 ASCII 문자가 들어 있습니다.  
문서를 처리하는 데 중요한 다양한 응용 분야가 있습니다. 스캐너나 팩스 파일의 점에서 개별 문자를 인식하는 것은 문자가 표준 패턴에 부합하는 한 오늘날 상당히 잘 수행되는 응용 분야 중 하나입니다.  
대규모 데이터베이스에서 인덱싱에 사용할 수 있는 정보의 의미론적 해석을 제공하는 것은 더 어려운 문제입니다.

### Accounting for snow cover using a satellite image
Much of the earth's surface is scanned regularly from satellites, which transmit their images to earth in digital form. These images can then be processed to extract a wealth of information. For example, inventory of the amount of snow in the watershed of a river may be critical for regulating a dam for flood control, water supply, or wildlife habitat. Estimates of snow mass can be made by accounting for the number of pixels in the image that appear as snow. A pixel from a satellite image might result from sensing a 10 meter by 10 meter spot of earth, but some satellites reportedly can see much smaller spots than that. Often, the satellite image must be compared to a map or other image to determine which pixels are in a particular area or watershed. This operation is usally manually-aided by a human user interacting with the image processing software and will be discussed more in Chapter 11 where image matching is covered. Figure 1.6 is a photograph taken on a space shuttle flight managed by the Johnson Space Center in Houston, Texas. It shows the town of Wenatchie, Washington, where the Wenatchie River flows into the Columbia River. Computers are known for their ability to handle large amounts of data; certainly the earth scanning satellites produce a tremendous amount of data useful for many purposes. For example, counts and locations of snow pixels might be input to a computer program that simulates the hydrology for that region. ( Temperature information for the region must be input to the program as well.) Another related application is taking inventory of crops and predicting harvests. Yet another is taking inventory of buildings for tax purposes: this is usually done manually with pictures taken from airplanes.

지구 표면의 많은 부분을 정기적으로 위성에서 스캔하여 이미지를 디지털 형태로 지구로 전송합니다.  
그런 다음 이러한 이미지를 처리하여 풍부한 정보를 추출할 수 있습니다.  
예를 들어, 강 유역의 눈 양을 재고하는 것은 홍수 조절, 상수도 또는 야생동물 서식지를 위한 댐을 규제하는 데 매우 중요할 수 있습니다.  
눈으로 보이는 이미지의 픽셀 수를 고려하여 눈의 질량을 추정할 수 있습니다.  
위성 이미지의 픽셀은 10미터 x 10미터 지점의 지구를 감지하면 발생할 수 있지만, 일부 위성은 그보다 훨씬 작은 지점을 볼 수 있다고 합니다.  
종종 위성 이미지를 지도 또는 기타 이미지와 비교하여 특정 지역이나 유역에 있는 픽셀을 결정해야 합니다.  
이 작업은 일반적으로 이미지 처리 소프트웨어와 상호 작용하는 인간 사용자가 수동으로 수행하며, 이미지 매칭에 대해서는 11장에서 자세히 설명합니다.  
그림 1.6은 텍사스 휴스턴의 존슨 우주 센터에서 관리하는 우주 왕복선 비행에서 촬영한 사진입니다.  
워싱턴주 웨나치에시 마을에서 웨나치 강이 컬럼비아 강으로 흘러드는 모습을 보여줍니다.  
컴퓨터는 대량의 데이터를 처리하는 능력으로 유명하며, 지구 스캔 위성은 여러 목적에 유용한 엄청난 양의 데이터를 생성합니다.  
예를 들어, 해당 지역의 수문을 시뮬레이션하는 컴퓨터 프로그램에 눈 픽셀의 개수와 위치를 입력할 수 있습니다. (이 지역의 온도 정보도 프로그램에 입력해야 합니다.)  
또 다른 관련 애플리케이션은 농작물 재고를 수집하고 수확량을 예측하는 것입니다.  
또 다른 애플리케이션은 세금 목적으로 건물 재고를 수집하는 것입니다: 일반적으로 비행기에서 촬영한 사진으로 수작업으로 이루어집니다.

### Understanding a scene of parts
At many points of manufacturing processes, parts are transferred on conveyors or in boxes. Parts must be individually placed in machines, packed, inspected, etc. If the operation is dull or dangerous, a vision-guided robot might provide a solution. The underlying image of Figure 1.7 shows three workpieces in a robot's workspace. By recognizing edges and holes, the robot vision system is able to guess at both the identity of a part and its position in the workspace. Using a 3D model made by computer-aided-design (CAD) for each guessed part and its guessed position, the vision system then compares the sensed image data with a computer graphic generated from the model and its position in space. Bad matches are rejected while good matches cause the guess to be refined. The bright lines in Figure 1.7 show three such refined matches between the image and models of the objects it contains. Finally, the robot eye-brain can tell the robot arm how to pick up a part and where to put it. The problems and techniques of 3D vision are covered in the later chapters of this text.

제조 공정의 많은 지점에서 부품은 컨베이어나 상자에 옮겨집니다.  
부품은 기계에 개별적으로 배치하거나 포장, 검사 등을 해야 합니다.  
작동이 느리거나 위험하다면 비전 가이드 로봇이 해결책을 제시할 수 있습니다.  
그림 1.7의 기본 이미지는 로봇의 작업 공간에 있는 세 개의 공작물을 보여줍니다.  
로봇 비전 시스템은 가장자리와 구멍을 인식하여 부품의 신원과 작업 공간 내 위치를 모두 추측할 수 있습니다.  
그런 다음 각 추측된 부품과 추측된 위치에 대해 컴퓨터 보조 설계(CAD)로 만든 3D 모델을 사용하여 비전 시스템은 감지된 이미지 데이터를 모델에서 생성된 컴퓨터 그래픽과 공간 내 위치를 비교합니다.  
일치하지 않는 경우에는 거부되지만 일치하는 경우에는 추측이 정교해집니다.  
그림 1.7의 밝은 선은 로봇이 포함된 물체의 이미지와 모델 사이에 세 가지 정교한 일치를 보여줍니다.  
마지막으로 로봇의 눈-뇌는 로봇 팔에 부품을 집어 드는 방법과 어디에 넣어야 하는지 알려줍니다. 3D 비전의 문제와 기술은 이 텍스트의 후반부에서 다룹니다.

<img width="592" alt="스크린샷 2025-04-14 오후 4 20 14" src="https://github.com/user-attachments/assets/da7ed989-05c6-4f49-b167-2497f08814ff" />

## 1.3 Operations on Images
This book presents a large variety of image operations. Operations can be grouped into different categories depending on their structure, level, or purpose. Some operations are for the purpose of improving the image solely for human consumption, while others are for extracting information for downstream automatic processing. Some operations create new output images, while others output non-image descriptions. A few important categories of image operations follow.

이 책에서는 다양한 이미지 작업을 소개합니다. 작업은 구조, 수준 또는 목적에 따라 다양한 범주로 그룹화할 수 있습니다.  
일부 작업은 사람이 소비하기 위한 목적으로만 이미지를 개선하기 위한 것이고, 다른 작업은 다운스트림 자동 처리를 위한 정보 추출을 위한 것입니다.  
일부 작업은 새로운 출력 이미지를 생성하는 반면, 다른 작업은 이미지가 아닌 설명을 출력하기도 합니다. 몇 가지 중요한 이미지 작업 범주는 다음과 같습니다.

### Changing pixels in small neighborhoods
Pixel values can be changed according to how they relate to a small number of neigh- boring pixels, for example, neighbors in adjacent rows or columns. Frequently, isolated 1's or O's in a binary image will be reversed in order to make them the same as their neighbors. The purpose of this operation could be to remove likely noise from the digitization process. Or, it could be just to simplify image content; for example, to ignore tiny islands in a lake or imperfections in a sheet of paper. Another common operation is to change border pixels to be background pixels as shown in Figure 1.8. The images of bacteria have fuzzy borders and often fuse together. By changing the black border pixels to white, the bacteria images, although smaller, have clearer borders and some formerly fusing pairs are separated. These operations are treated in Chapter 3.

픽셀 값은 인접한 행이나 열의 이웃과 같은 소수의 인접 픽셀과 어떻게 관련되는지에 따라 변경할 수 있습니다.  
이진 이미지에서 고립된 1 또는 O는 이웃 픽셀과 동일하게 만들기 위해 반전되는 경우가 많습니다.  
이 작업의 목적은 디지털화 프로세스에서 발생할 수 있는 노이즈를 제거하는 것일 수 있습니다.  
또는 이미지 콘텐츠를 단순화하는 것일 수도 있습니다.  
예를 들어 호수의 작은 섬이나 종이 한 장의 불완전성을 무시하는 것과 같이 이미지 콘텐츠를 단순화하는 것일 수도 있습니다.  
또 다른 일반적인 작업은 그림 1.8과 같이 경계 픽셀을 배경 픽셀로 변경하는 것입니다.  
박테리아의 이미지는 경계가 흐릿하고 종종 서로 융합됩니다.  
검은색 경계 픽셀을 흰색으로 변경하면 박테리아 이미지는 작지만 더 선명한 경계를 가지며 이전에는 융합된 쌍이 분리됩니다. 이러한 작업은 3장에서 다룹니다.

<img width="592" alt="스크린샷 2025-04-14 오후 4 23 25" src="https://github.com/user-attachments/assets/73a20930-c411-4200-bd87-49227329b85c" />

### Enhancing an entire image
Some operations treat the entire image in a uniform manner. The image might be too dark - say its maximum brightness value is 120 - so all brightness values can be scaled up by a factor of 2 to improve its displayed appearance. Noise or unnecessary detail can be removed by replacing the value of every input pixel with the average of all nine pixels in its immediate neighborhood. Alternatively, details can be enhanced by replacing each pixel value by the contrast between it and its neighbors. Figure 1.9 shows a simple contrast com- putation applied at all pixels of an input image. Note how the boundaries of most objects are well detected. The output image results from computations made only on the local 3x3 neighborhoods of the input image. Chapter 5 describes several of these kinds of operations. Perhaps an image is taken using a fish eye lens and we want to create an output image with less distortion: in this case, we have to "move" the pixel values to other locations in the image to move them closer to the image center. Such an operation is called image warping and is covered in Chapter 11.

일부 작업은 이미지 전체를 균일하게 처리합니다.  
이미지가 너무 어두울 수 있으므로 최대 밝기 값이 120이라고 가정하면 모든 밝기 값을 2배로 확대하여 표시되는 모양을 개선할 수 있습니다.  
노이즈나 불필요한 세부 사항은 각 입력 픽셀의 값을 해당 인접한 모든 픽셀의 평균으로 대체하여 제거할 수 있습니다.  
또는 각 픽셀 값을 인접한 픽셀과 인접한 픽셀 간의 대비로 대체하여 세부 사항을 강화할 수도 있습니다.  
그림 1.9는 입력 이미지의 모든 픽셀에 적용된 간단한 대비 계산을 보여줍니다.  
대부분의 객체의 경계가 잘 감지되는 방식을 주목하세요.  
출력 이미지는 입력 이미지의 로컬 3x3 이웃에서만 수행된 계산에서 결과를 얻습니다.  

<img width="592" alt="스크린샷 2025-04-14 오후 4 26 19" src="https://github.com/user-attachments/assets/c531d6b9-3dba-4dd8-ad65-a7610714b623" />

5장에서는 이러한 종류의 작업 중 몇 가지를 설명합니다.  
아마도 어안 렌즈를 사용하여 이미지를 촬영하여 왜곡이 적은 출력 이미지를 만들고 싶을 수도 있습니다:  
이 경우 픽셀 값을 이미지 중심에 더 가깝게 이동하려면 이미지의 다른 위치로 "이동"해야 합니다. 이러한 작업을 이미지 왜곡이라고 하며 11장에서 다룹니다.

### Combining multiple images
An image can be created by adding or subtracting two input images. Image subtraction is commonly used to detect change over time. Figure 1.10 shows two images of a moving part and the difference image resulting from subtracting the corresponding pixel values of the second image from those of the first image. Image subtraction captures the boundary of the moving object, but not perfectly. ( Since negative pixel values were not used, not all changes were saved in the output image.) In another application, urban development might be more easily seen by subtracting an aerial image of a city taken five years ago from a current image of the city. Image addition is also useful. Figure 1.11 shows an image of Thomas Jefferson "added" to an image of the great arch opening onto the lands of the Louisiana Purchase; more work is needed in this case to blend the images better.

이미지는 두 개의 입력 이미지를 더하거나 빼는 방식으로 만들 수 있습니다.  
이미지 뺄셈은 일반적으로 시간 경과에 따른 변화를 감지하는 데 사용됩니다.  
그림 1.10은 움직이는 부분의 두 이미지와 첫 번째 이미지에서 두 번째 이미지의 해당 픽셀 값을 뺀 차이 이미지를 보여줍니다.  

<img width="592" alt="스크린샷 2025-04-14 오후 4 27 28" src="https://github.com/user-attachments/assets/5c79c995-0757-4572-8e44-332b7d2c79eb" />

이미지 뺄셈은 움직이는 물체의 경계를 포착하지만 완벽하지는 않습니다. (음의 픽셀 값이 사용되지 않았기 때문에 모든 변경 사항이 출력 이미지에 저장된 것은 아닙니다.)  
다른 응용 프로그램에서는 현재 도시 이미지에서 5년 전에 촬영한 도시의 항공 이미지를 빼면 도시 개발을 더 쉽게 볼 수 있습니다.  
이미지 추가도 유용합니다. 그림 1.11은 루이지애나 매입 부지에 있는 대아치 개구부 이미지에 토마스 제퍼슨이 "추가"된 이미지를 보여줍니다.  

<img width="592" alt="스크린샷 2025-04-14 오후 4 28 14" src="https://github.com/user-attachments/assets/56e00ef7-daf8-4520-9149-c5d38fb7f675" />

이 경우 이미지를 더 잘 블렌딩하기 위해 더 많은 작업이 필요합니다.

We have already seen the example of counting holes. More generally, the regions of 0's corresponding to holes in the crossbar inspection problem could be images of objects, often called blobs - perhaps these are microbes in a water sample. Important features might be average object area, perimeter, direction, etc. We might want to output these important features separately for every detected object. Chapter 3 describes such processing. Chapters 6 and 7 discuss means of quantitatively summarizing the color or texture content of regions of an image. Chapter 4 shows how to classify objects according to these features; for example, is the extracted region the image of microbe A or B? Figure 1.12 shows output from a well-known algorithm applied to the bacteria image of Figure 1.8 giving features of separate regions identified in the image, including the region area and location. Regions with area of a few hundred pixels correspond to isolated bacteria while the large region is due to several touching bacteria.

우리는 이미 구멍을 세는 예를 보았습니다. 더 일반적으로 크로스바 검사 문제에서 구멍에 해당하는 0의 영역은 종종 블롭(얼룩)이라고 불리는 물체의 이미지일 수 있으며, 아마도 이는 물 샘플의 미생물일 수 있습니다.  
중요한 특징은 평균 물체 면적, 둘레, 방향 등일 수 있습니다. 이러한 중요한 특징은 감지된 모든 물체에 대해 개별적으로 출력하는 것이 좋습니다.  
3장에서는 이러한 처리에 대해 설명합니다. 6장과 7장에서는 이미지 영역의 색상 또는 질감 내용을 정량적으로 요약하는 방법에 대해 설명합니다. 4장에서는 이러한 특징에 따라 물체를 분류하는 방법을 보여줍니다.  
예를 들어, 추출된 영역이 미생물 A 또는 B의 이미지인가요?  
그림 1.12는 그림 1.8의 박테리아 이미지에 적용된 잘 알려진 알고리즘의 출력을 보여주며, 이미지에서 식별된 개별 영역의 영역과 위치를 포함한 특징을 제공합니다.  

<img width="592" alt="스크린샷 2025-04-14 오후 4 29 59" src="https://github.com/user-attachments/assets/d24e1ef7-ef4b-49ef-9264-e1df58968806" />

면적이 수백 픽셀인 영역은 분리된 박테리아에 해당하는 반면, 큰 영역은 여러 접촉하는 박테리아에 의한 것입니다.

### Extracting non-iconic representations
Higher-level operations usually extract representations of the image that are non-iconic, that is, data structures that are not like an image. (Recall that extraction of such descrip- tions is often defined to be the goal of computer vision.) Figure 1.12 shows a non-iconic description derived from the bacteria image. In addition to examples already mentioned, consider a report of the count of microbes of type A and B in a slide from a microscope or the volume of traffic flow between two intersections of a city computed from a video taken from a utility pole. In another important application, the (iconic) input might be a scanned magazine article and the output a hypertext structure containing sections of recognized ASCII text and sections of raw images for the figures. As a final example, in the application illustrated in Figure 1.7, the machine vision system would output a set of three detections, each encoding a part number, three parameters of part position and three parameters of the orientation of the part. This scene description could then be turned over to the motion- planning system, which would decide on how to manipulate the three parts.

상위 수준의 작업은 일반적으로 이미지가 아닌 이미지의 표현, 즉 이미지와 같지 않은 데이터 구조를 추출합니다. (이러한 설명의 추출은 종종 컴퓨터 비전의 목표로 정의된다는 점을 상기하십시오.)  
그림 1.12는 박테리아 이미지에서 파생된 비아이콘 설명을 보여줍니다.  
이미 언급한 예 외에도 현미경 슬라이드에서 A형과 B형 미생물의 수 또는 전신주에서 촬영한 비디오에서 계산된 도시의 두 교차로 사이의 교통량에 대한 보고서를 고려하십시오.  
또 다른 중요한 애플리케이션에서는 (상징적인) 입력 데이터가 스캔된 잡지 기사일 수 있으며, 인식된 ASCII 텍스트 섹션과 도형에 대한 원시 이미지 섹션을 포함하는 하이퍼텍스트 구조를 출력할 수 있습니다.  
마지막 예로, 그림 1.7에 도시된 애플리케이션에서 머신 비전 시스템은 각각 부품 번호, 부품 위치의 및 부품 방향의 세 가지 매개변수를 인코딩하는 세 가지 감지 세트를 출력합니다.  
이 장면 설명은 모션 플래닝 시스템으로 넘겨져 세 가지 부품을 조작하는 방법을 결정할 수 있습니다.

## 1.4 The Good, the Bad, and the Ugly
Having cited many applications of machine vision, we cannot proceed without saying that success is usually hard won. Often, implementors have to accept environmental constraints that compromise system flexibility. For example, scene lighting might have to be carefully controlled, or objects might have to be mechanically separated or positioned before imaging. This is because the real world yields exorbitant variations in the input image, challenging the best computer algorithms in their task of extracting the "essence" or invariant features of objects. Appearance of an object can vary significantly due to changes in illumination or presence of other objects, which might be unexpected. Consider, for example, the shadows in Figure 1.7 and Figure 1.9. Moreover, decisions about object structure must often be made by integrating a variety of information from many pixels of the image. For example, the brightness of the tops of the glasses on the counter in Figure 1.9 is the same as that of the wall, so no glass-wall boundary is evident at the pixel level. In order to recognize each glass as a separate object, pixels from a wider area must be grouped and organized.

머신 비전의 많은 응용 분야를 인용한 결과, 성공은 일반적으로 어려운 일이라는 말을 하지 않고는 진행할 수 없습니다.  
종종 구현자는 시스템의 유연성을 저해하는 환경적 제약을 받아들여야 합니다.  
예를 들어 장면 조명을 신중하게 제어해야 하거나 이미지를 촬영하기 전에 물체를 기계적으로 분리하거나 위치시켜야 할 수도 있습니다.  
이는 현실 세계가 입력 이미지에 과도한 변화를 일으켜 물체의 '본질' 또는 불변 특징을 추출하는 작업에서 최고의 컴퓨터 알고리즘에 도전하기 때문입니다.  
물체의 모양은 조명이나 다른 물체의 존재 변화로 인해 크게 달라질 수 있으며, 이는 예상치 못한 일일 수 있습니다.  
예를 들어 그림 1.7과 그림 1.9의 그림자를 생각해 보세요.  
또한 물체 구조에 대한 결정은 이미지의 여러 픽셀에서 얻은 다양한 정보를 통합하여 내려야 하는 경우가 많습니다.  
예를 들어 그림 1.9의 카운터에 있는 안경 상단의 밝기는 벽의 밝기와 동일하므로 픽셀 수준에서 유리벽 경계가 명확하지 않습니다.  
각 유리를 별도의 물체로 인식하려면 더 넓은 영역의 픽셀을 그룹화하여 정리해야 합니다.

Humans are quite good at this, but developing flexible grouping processes for machine vision has proved difficult. Problems of occlusion hamper recognition of 3D objects. Can a vision system recognize the person or the chair in Figure 1.9, even though neither appears to have legs? At a higher level yet, what model of a dog could empower a machine to recognize the diverse individuals that could be imaged? These difficulties, and others, will be discussed further throughout this book.

인간은 이 분야에 꽤 능숙하지만 기계 비전을 위한 유연한 픽셀 그룹화 프로세스를 개발하는 것은 어려운 일입니다.  
가림 문제는 3D 물체의 인식을 방해합니다.  
그림 1.9에서 두 물체 모두 다리가 없는 것처럼 보이지만 비전 시스템이 사람이나 의자를 인식할 수 있을까요?  
아직 더 높은 수준에서 어떤 모델의 개가 기계가 이미지화할 수 있는 다양한 개체를 인식할 수 있도록 힘을 실어줄 수 있을까요?  
이러한 어려움 등에 대해서는 이 책 전반에 걸쳐 자세히 설명하겠습니다.

## 1.5 Use of Computers and Software
Computers are legendary for accurate accounting of quantitative information. Computing with images has gone on for over 30 years - initially mostly in research labs with mainframe computers or in production shops with special-purpose computers. Recently, large inexpen- sive memories and high speed general-purpose processors have brought image computing potential to every multimedia personal computer user, including the hobbyist working in her dining room.

컴퓨터는 정량적 정보를 정확하게 설명하는 것으로 전설적인 존재입니다.  
이미지를 이용한 컴퓨팅은 30년 넘게 이어져 왔습니다.  
처음에는 주로 메인프레임 컴퓨터가 있는 연구실이나 특수 목적 컴퓨터가 있는 생산 공장에서 이루어졌습니다.  
최근에는 대형 저가 메모리와 고속 범용 프로세서를 통해 식당에서 일하는 취미자를 포함한 모든 멀티미디어 개인용 컴퓨터 사용자에게 이미지 컴퓨팅의 잠재력을 제공하고 있습니다.


One can compute with images in different ways. The easiest is to acquire an existing program that can perform many of the needed image operations. Some programs are free to the public; others must be purchased: some options are given in the Appendices. Many free images are available from the World-Wide-Web. To control your own image input, you can buy a flatbed scanner or a digital camera, each available for a few hundred dollars. Software libraries are available which contain many subroutines for processing images: the user writes an application program which calls the library routines to perform the required operations on the user's image data. Most companies selling input devices for machine vision also provide libraries for image operations and even driver programs with nice graphical user interfaces (GUI). Special-purpose hardware is available for speeding up image operations that can take many seconds, or even minutes, on a general purpose processor. Many of the early parallel computers costing millions of dollars were designed with image processing as a primary task; however, today most of the critical operations can be provided by sets of boards costing a few thousand dollars. Usually, special hardware is only needed for high production rates or real-time response. Special programming languages with images and image operations as language primitives have been defined; sometimes, these have been combined with operations for controlling an industrial robot. Today, it is apparent that much good image processing can and will be done using a general purpose language, such as C, and a general purpose computer available via mail order or the local computer store. This bodes exceedingly well for the machine vision field, since challenging problems will now be attacked from all directions possible! The reader is invited to join in.

이미지를 사용하여 다양한 방식으로 계산할 수 있습니다.  
가장 쉬운 방법은 필요한 많은 이미지 작업을 수행할 수 있는 기존 프로그램을 구입하는 것입니다.  
일부 프로그램은 일반 대중에게 무료로 제공되며, 다른 프로그램은 구매해야 합니다. 일부 옵션은 부록에서 제공됩니다.  
많은 무료 이미지는 World-Wide-Web에서 이용할 수 있습니다.  
이미지 입력을 제어하려면 플랫베드 스캐너나 디지털 카메라를 구입할 수 있으며, 각각 몇 백 달러에 구입할 수 있습니다.  
이미지 처리를 위한 많은 하위 루틴이 포함된 소프트웨어 라이브러리를 이용할 수 있습니다:  
사용자는 라이브러리 루틴을 호출하여 사용자의 이미지 데이터에 필요한 작업을 수행하는 애플리케이션 프로그램을 작성합니다.  
머신 비전용 입력 장치를 판매하는 대부분의 회사는 이미지 작업을 위한 라이브러리를 제공하며, 심지어 멋진 그래픽 사용자 인터페이스(GUI)를 갖춘 드라이버 프로그램도 제공합니다.  
범용 프로세서에서 이미지 작업 속도를 높이기 위해 특수 목적 하드웨어가 사용됩니다.  
수백만 달러가 드는 초기 병렬 컴퓨터 중 상당수는 이미지 처리를 주요 작업으로 설계되었지만, 오늘날 대부분의 중요한 작업은 몇 천 달러가 드는 보드 세트를 통해 제공할 수 있습니다.  
일반적으로 특수 하드웨어는 높은 생산 속도나 실시간 응답에만 필요합니다.  
이미지와 이미지 작업을 기본 언어로 하는 특수 프로그래밍 언어가 정의되었으며, 때로는 산업용 로봇을 제어하기 위한 작업과 결합되기도 합니다.  
오늘날 C와 같은 범용 언어와 우편 주문 또는 로컬 컴퓨터 스토어를 통해 사용할 수 있는 범용 컴퓨터를 사용하여 많은 좋은 이미지 처리를 할 수 있고 앞으로도 가능할 것입니다.  
이는 머신 비전 분야에서 매우 좋은 징조이며, 이제 어려운 문제가 모든 방향에서 공격받을 수 있기 때문입니다! 독자도 참여할 것을 초대합니다.

## 1.6 Related Areas
Computer vision is related to many other disciplines: we are not able to pursue all of these relations in depth in this text. First, it is important to distinguish between image processing and image understanding. Image processing is primarily concerned with the transformation of images into images, whereas, image understanding is concerned with making decisions based on images and explicitly constructing the scene descriptions needed to do so. Image processing is quite often used in support of image understanding and thus will be treated to some extent in this book. Books concerned with image processing typically are based on the model of an image as a continuous function f(x, y) of two spatial parameters x and y, whereas this text will concentrate on the model of an image as a discrete 2D array I r , c of integer brightness samples. In this book, we use the terms computer vision, machine vision, and image understanding interchangeably; however, experts would certainly debate their nuances.

컴퓨터 비전은 다른 많은 분야와 관련이 있습니다. 이 텍스트에서는 이러한 모든 관계를 깊이 있게 추구할 수 없습니다.  
첫째, 이미지 처리와 이미지 이해를 구분하는 것이 중요합니다.  
이미지 처리는 주로 이미지를 이미지로 변환하는 것과 관련이 있는 반면, 이미지 이해는 이미지를 기반으로 의사 결정을 내리고 이를 위해 필요한 장면 설명을 명시적으로 구성하는 것과 관련이 있습니다.  
이미지 처리는 이미지 이해를 지원하는 데 자주 사용되므로 이 책에서는 어느 정도 다룰 것입니다.  
이미지 처리와 관련된 책들은 일반적으로 두 공간 매개변수 x와 y의 연속 함수 f(x, y)로서 이미지 모델을 기반으로 하는 반면, 이 글에서는 정수로 된 밝기 샘플의 이산 2D 배열 I[r, c] 로서 이미지 모델을 중심으로 다룹니다.  
이 책에서는 컴퓨터 비전, 머신 비전, 이미지 이해라는 용어를 혼용하여 사용하지만, 전문가들은 이들의 뉘앙스에 대해 확실히 논의할 것입니다.

The psychology of human perception is very important for two reasons; first, the creator of images for human consumption must be aware of the characteristics of the client, and secondly, study of the tremendous human capability in image understanding can guide our development of algorithms. While this text includes some discussion of human perception and cognition, its approach is primarily hands-on problem solving. The physics of light, including optics and color science, is important to our study. We will present the basic material necessary; however, readers who want to be experts on illumination, sensing, or lenses will need to access the related literature. A variety of mathematical models are used throughout the text; for mastery, the reader must be comfortable with the notions of functions, probability, calculus and analytical geometry. The intuitive concepts of image processing often strengthen the mathematical concepts. Finally, any book about computer vision must be strongly related to computer graphics. Both fields are concerned with how objects are viewed and how objects are modeled; the prime distinction is one of direction - computer vision is concerned with description and recognition of objects from images, while computer graphics is concerned with generation of images from object descriptions. Recently, there has been a great deal of integration of these two areas: computer graphics is needed to display computer vision results and computer vision is needed to make object models. Digital images are commonly used as input for computer graphics products.

인간 인식의 심리학은 두 가지 이유로 매우 중요합니다.  
첫째, 인간 소비를 위한 이미지를 만드는 사람은 고객의 특성을 인식해야 하며, 둘째, 이미지 이해에서 인간의 엄청난 능력에 대한 연구는 알고리즘 개발에 도움이 될 수 있습니다.  
이 텍스트에는 인간의 인식과 인지에 대한 논의가 일부 포함되어 있지만, 그 접근 방식은 주로 실습 문제 해결입니다.  
광학과 색채 과학을 포함한 빛의 물리학은 우리 연구에 중요합니다.  
필요한 기본 자료를 제시하겠지만, 조명, 센싱 또는 렌즈 전문가가 되고 싶은 독자는 관련 문헌에 접근해야 합니다.  
텍스트 전반에 걸쳐 다양한 수학적 모델이 사용되며, 숙달을 위해서는 독자가 함수, 확률, 미적분 및 분석 기하학의 개념에 익숙해야 합니다.  
이미지 처리에 대한 직관적인 개념은 종종 수학적 개념을 강화합니다.  
마지막으로, 컴퓨터 비전에 관한 모든 책은 컴퓨터 그래픽과 밀접한 관련이 있어야 합니다.  
두 분야 모두 사물을 보는 방식과 사물을 모델링하는 방식에 중점을 둡니다.  
가장 큰 차이점은 방향 중 하나로, 컴퓨터 비전은 이미지에서 사물을 설명하고 인식하는 방식에 중점을 두고 있으며, 컴퓨터 그래픽은 사물 설명에서 이미지를 생성하는 방식에 중점을 둡니다.  
최근에는 이 두 분야가 크게 통합되었습니다: 컴퓨터 비전 결과를 표시하기 위해 컴퓨터 그래픽이 필요하고 물체 모델을 만들기 위해 컴퓨터 비전이 필요합니다.  
디지털 이미지는 일반적으로 컴퓨터 그래픽 제품의 입력으로 사용됩니다.

## 1.7 The rest of the book
The previous sections informally introduced many of the concepts in the book and indicated the chapters in which they are treated. The reader should now appreciate the range of problems attacked by machine vision and a few of its methods. The chapters that im- mediately follow describe 2D machine vision. In those chapters, the image is analyzed in self-referencing terms of pixels, rows, intersections, colors, textures, etc. To be sure, knowl- edge about how the image was taken from the real 3D world is present, but the relationship between image pixels and real-world elements is obvious - only the scale is different. For example, a radiologist can readily tell from an image if a blood vessel is constricted without knowing much about the physics of the sensor or about what portion of the body a pixel represents. So can a machine vision program. Similarly, the essence of a character recog- nition algorithm has nothing to do with the real font size being scanned. Consequently, the material in Chapters 2 to 1 has a 2D character and is more generic and simpler than material in Chapters 12 to 15. In Chapters 12 to 15, the 3D nature of objects and the view- points used to image them are crucial. The analysis cannot be done with the coordinates of a single image because we need to relate multiple images or images and models; or, we need to relate a sensor's view to a robot's view. In Chapters 12 to 15, we are analyzing 3D scenes, not 2D images, and the most important tool for the analysis is 3D analytical geometry. As in computer graphics, the step from 2D to 3D is a large one in terms of both modeling abstraction and computational effort.

이전 섹션에서는 책의 많은 개념을 비공식적으로 소개하고 그 개념을 다루는 챕터를 표시했습니다.  
이제 독자는 머신 비전과 그 방법 중 몇 가지가 공격하는 다양한 문제의 범위를 이해해야 합니다. 바로 이어지는 챕터에서는 2D 머신 비전을 설명합니다.  
해당 챕터에서는 이미지를 픽셀, 행, 교차점, 색상, 텍스처 등의 자기 참조 용어로 분석합니다.  
실제 3D 세계에서 이미지를 가져온 방식에 대한 지식은 있지만 이미지 픽셀과 실제 요소 간의 관계는 분명합니다 - 규모만 다를 뿐입니다.  
예를 들어 방사선 전문의는 센서의 물리학이나 픽셀이 신체의 어떤 부분을 나타내는지에 대해 잘 알지 못한 채 혈관이 수축했는지 이미지를 통해 쉽게 알 수 있습니다. 따라서 머신 비전 프로그램도 마찬가지입니다.  
마찬가지로 문자 인식 알고리즘의 본질은 스캔되는 실제 글꼴 크기와는 아무런 관련이 없습니다.  
따라서 2장에서 1장까지의 자료는 2D 캐릭터를 가지고 있으며 12장에서 15장의 자료보다 더 일반적이고 간단합니다.  
12장에서 15장까지는 물체의 3D 특성과 이미지를 이미징하는 데 사용되는 뷰 포인트가 중요합니다.  
여러 이미지 또는 이미지와 모델을 연관시켜야 하므로 단일 이미지의 좌표로는 분석을 할 수 없거나 센서의 뷰를 로봇의 뷰와 연관시켜야 합니다.  
12장에서 15장까지는 2D 이미지가 아닌 3D 장면을 분석하고 있으며, 가장 중요한 분석 도구는 3D 분석 기하학입니다.  
컴퓨터 그래픽과 마찬가지로 2D에서 3D로의 단계는 모델링 추상화와 계산 노력 측면에서 모두 큰 단계입니다.





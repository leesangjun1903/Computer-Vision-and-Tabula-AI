# Vision, the challenge
## 1.1 INTRODUCTION—MAN AND HIS SENSES
Of the five senses—vision, hearing, smell, taste, and touch—vision is undoubtedly the one that man has come to depend upon above all others, and indeed the one that provides most of the data he receives. Not only do the input pathways from the eyes provide megabits of information at each glance but also the data rates for continuous viewing probably exceed 10 Mbps. However, much of this information is redundant and is compressed by the various layers of the visual cortex, so that the higher centers of the brain have to interpret abstractly only a small fraction of the data. Nonetheless, the amount of information the higher centers receive from the eyes must be at least two orders of magnitude greater than all the information they obtain from the other senses.

Another feature of the human visual system is the ease with which interpretation is carried out. We see a scene as it is—trees in a landscape, books on a desk, widgets in a factory. No obvious deductions are needed and no overt effort is required to interpret each scene; in addition, answers are effectively immediate and are normally available within a tenth of a second. Just now and again some doubt arises—e.g., a wire cube might be “seen” correctly or inside out. This and a host of other optical illusions are well known, although for the most part we can regard them as curiosities—irrelevant freaks of nature. Somewhat surprisingly, illusions are quite important, since they reflect hidden assumptions that the brain is making in its struggle with the huge amounts of complex visual data it is receiving. We have to pass by this story here (although it resurfaces now and again in various parts of this book). However, the important point is that we are for the most part unaware of the complexities of vision. Seeing is not a simple process: it is just that vision has evolved over millions of years, and there was no particular advantage in evolution giving us any indication of the difficulties of the task (if anything, to have done so would have cluttered our minds with irrelevant information and slowed our reaction times).

시각, 청각, 후각, 미각, 촉각 등 다섯 가지 감각 중 시각은 의심할 여지 없이 인간이 무엇보다도 의존하게 된 감각이며, 실제로 인간이 받는 대부분의 데이터를 제공하는 감각입니다.  
눈의 입력 경로는 각 눈에 메가비트의 정보를 제공할 뿐만 아니라 연속 시청을 위한 데이터 전송 속도도 10Mbps를 초과할 수 있습니다.  
그러나 이러한 정보의 대부분은 중복되어 시각 피질의 다양한 층에 의해 압축되므로 뇌의 상위 중추는 데이터의 극히 일부만 추상적으로 해석해야 합니다.  
그럼에도 불구하고 상위 중추가 눈으로부터 받는 정보의 양은 다른 감각으로부터 얻는 모든 정보보다 최소 두 자릿수 이상 커야 합니다.

인간 시각 시스템의 또 다른 특징은 해석이 쉽게 이루어진다는 점입니다.  
우리는 풍경 속의 나무, 책상 위의 책, 공장의 위젯 등 장면을 있는 그대로 봅니다.  
각 장면을 해석하는 데 명확한 추론이 필요하지 않으며, 명확한 노력도 필요하지 않습니다.  
또한 답변은 효과적으로 즉각적이며 일반적으로 10분의 1초 이내에 이용할 수 있습니다.  
방금 몇 번이고 몇 가지 의문이 제기되었는데, 예를 들어 와이어 큐브가 "정확하게" 보이거나 안쪽으로 보이는 것과 같은 의문이 제기되었습니다.  
이와 다른 여러 착시 현상들은 잘 알려져 있지만, 대부분의 경우 우리는 이를 호기심으로 간주할 수 있습니다.  
하지만 이는 뇌가 수신하는 방대한 양의 복잡한 시각 데이터로 인해 어려움을 겪고 있다는 숨겨진 가정을 반영하기 때문에 다소 놀랍게도 착시 현상은 매우 중요합니다.  
우리는 여기서 이 이야기를 지나쳐야 합니다(비록 이 이야기는 이 책의 여러 부분에서 반복적으로 다시 등장하지만).  
하지만 중요한 점은 우리가 시각의 복잡성을 대부분 인식하지 못하고 있다는 것입니다.  
보는 것은 간단한 과정이 아닙니다. 단지 시각이 수백만 년에 걸쳐 진화해왔고, 진화 과정에서 특별한 이점이 없었기 때문에 과제의 어려움을 나타내는 데 아무런 도움이 되지 않았을 뿐입니다(그렇게 했다면 관련 없는 정보로 마음을 어지럽히고 반응 시간을 늦췄을 것입니다).

In the present-day and age, man is trying to get machines to do much of his work for him. For simple mechanistic tasks this is not particularly difficult, but for more complex tasks the machine must be given the sense of vision. Efforts have been made to achieve this, sometimes in modest ways, for well over 40 years. At first, schemes were devised for reading, for interpreting chromosome images, and so on; but when such schemes were confronted with rigorous practical tests, the problems often turned out to be more difficult. Generally, researchers react to finding that apparent “trivia” are getting in the way by intensifying their efforts and applying great ingenuity, and this was certainly so with early efforts at vision algorithm design. However, it soon became plain that the task really is a complex one, in which numerous fundamental problems confront the researcher, and the ease with which the eye can interpret scenes turned out to be highly deceptive. 

Of course, one of the ways in which the human visual system gains over the machine is that the brain possesses more than 1010 cells (or neurons), some of which have well over 10,000 contacts (or synapses) with other neurons. If each neuron acts as a type of microprocessor, then we have an immense computer in which all the processing elements can operate concurrently. Taking the largest single man-made computer to contain several hundred million rather modest processing elements, the majority of the visual and mental processing tasks that the eye brain system can perform in a flash have no chance of being performed by present-day man-made systems. Added to these problems of scale, there is the problem of how to organize such a large processing system and also how to program it. Clearly, the eye brain system is partly hard-wired by evolution but there is also an interesting capability to program it dynamically by training during active use. This need for a large parallel processing system with the attendant complex control problems shows that computer vision must indeed be one of the most difficult intellectual problems to tackle.

현대와 시대에 인간은 기계가 자신의 많은 작업을 대신 수행하도록 하려고 노력하고 있습니다.  
단순한 기계적 작업의 경우 이는 특별히 어렵지 않지만, 더 복잡한 작업의 경우 기계에 시각을 부여해야 합니다.  
이를 달성하기 위한 노력은 40년이 훨씬 넘는 기간 동안 때로는 겸손한 방식으로 이루어졌습니다.  
처음에는 염색체 이미지를 읽고 해석하는 등의 방법이 고안되었지만, 이러한 방법이 엄격한 실제 테스트에 직면했을 때 문제는 종종 더 어려워졌습니다.  
일반적으로 연구자들은 명백한 '트라이비아(잡학)'가 자신의 노력을 강화하고 뛰어난 독창성을 발휘함으로써 방해가 되고 있다는 사실에 반응하며, 이는 비전 알고리즘 설계의 초기 노력에서도 마찬가지였습니다.  
그러나 이 작업은 연구자가 직면한 수많은 근본적인 문제와 눈으로 장면을 해석할 수 있는 용이성이 매우 기만적이라는 것이 곧 분명해졌습니다. 

물론 인간의 시각 시스템이 기계를 능가하는 방법 중 하나는 뇌가 1010개 이상의 세포(또는 뉴런)를 가지고 있으며, 그 중 일부는 다른 뉴런과 10,000개 이상의 접촉(또는 시냅스)을 가지고 있다는 것입니다.  
각 뉴런이 일종의 마이크로프로세서 역할을 한다면 모든 처리 요소가 동시에 작동할 수 있는 거대한 컴퓨터가 있습니다.  
가장 큰 단일 인공 컴퓨터가 수억 개의 다소 소박한 처리 요소를 포함하고 있기 때문에 눈 뇌 시스템이 순식간에 수행할 수 있는 시각적 및 정신적 처리 작업의 대부분은 오늘날의 인공 시스템에서 수행될 가능성이 없습니다.  
이러한 규모의 문제 외에도 이러한 대규모 처리 시스템을 어떻게 구성하고 프로그래밍해야 하는지에 대한 문제도 있습니다.  
분명히 눈 뇌 시스템은 부분적으로 진화에 의해 연결되어 있지만 능동적으로 사용하는 동안 훈련하여 동적으로 프로그래밍할 수 있는 흥미로운 기능도 있습니다.  
복잡한 제어 문제가 수반되는 대규모 병렬 처리 시스템의 필요성은 컴퓨터 비전이 실제로 해결하기 가장 어려운 지적 문제 중 하나임을 보여줍니다.

So what are the problems involved in vision that make it apparently so easy for the eye, yet so difficult for the machine? In the next few sections an attempt is made to answer this question.

그렇다면 시각과 관련된 문제 중 눈에는 그렇게 쉽지만 기계에는 그렇게 어려운 것은 무엇일까요?  
다음 몇 섹션에서는 이 질문에 답하기 위해 시도합니다.

## 1.2 THE NATURE OF VISION
### 1.2.1 THE PROCESS OF RECOGNITION
This section illustrates the intrinsic difficulties of implementing computer vision, starting with an extremely simple example—that of character recognition. Consider the set of patterns shown in Fig. 1.1A. Each pattern can be considered as a set of 25 bits of information, together with an associated class indicating its interpretation. In each case imagine a computer learning the patterns and their classes by rote. Then any new pattern may be classified (or “recognized”) by comparing it with this previously learnt “training set,” and assigning it to the class of the nearest pattern in the training set. Clearly, test pattern (1) (Fig. 1.1B) will be allotted to class U on this basis. Chapter 13, Basic Classification Concepts, shows that this method is a simple form of the nearest neighbor approach to pattern recognition.

이 섹션은 컴퓨터 비전 구현의 본질적인 어려움을 매우 간단한 예인 문자 인식부터 설명합니다.  
그림 1.1A에 표시된 패턴 집합을 고려해 보세요. 각 패턴은 25비트의 정보 집합과 그 해석을 나타내는 관련 클래스로 간주될 수 있습니다.  

<img width="689" alt="스크린샷 2025-04-15 오전 12 12 19" src="https://github.com/user-attachments/assets/8294bc6a-f0ec-4149-9731-f5930481d3dd" />

각 경우에 컴퓨터가 패턴과 그 클래스를 루트별로 학습한다고 상상해 보세요.  
그런 다음 새로운 패턴을 이전에 학습된 "훈련 세트"와 비교하여 이를 훈련 세트의 가장 가까운 패턴 클래스에 할당하여 분류(또는 "인식")할 수 있습니다.  
분명히 테스트 패턴 (1) (그림 1.1B)은 이를 기준으로 클래스 U에 할당됩니다. 

<img width="719" alt="스크린샷 2025-04-15 오전 12 13 18" src="https://github.com/user-attachments/assets/7488b164-0eeb-4336-84b9-31ae28e2079b" />

13장 기본 분류 개념에서는 이 방법이 패턴 인식에 대한 가장 가까운 이웃 접근 방식의 간단한 형태임을 보여줍니다.

The scheme outlined above seems straightforward and is indeed highly effective, even being able to cope with situations where distortions of the test patterns occur or where noise is present: this is illustrated by test patterns (2) and (3). However, this approach is not always foolproof. First, there are situations where distortions or noise is excessive, so errors of interpretation arise. Second, there are situations where patterns are not badly distorted or subject to obvious noise, yet are misinterpreted: this seems much more serious, since it indicates an unexpected limitation of the technique rather than a reasonable result of noise or distortion. In particular, these problems arise where the test pattern is displaced or misorientated relative to the appropriate training set pattern, as with test pattern (6). 

As will be seen in Chapter 13, Basic Classification Concepts, there is a power- ful principle that indicates why the unlikely limitation given above can arise: it is simply that there are insufficient training set patterns, and that those that are present are insufficiently representative of what will arise in practical situations. Unfortunately, this presents a major difficulty, since providing enough training set patterns incurs a serious storage problem and an even more serious search prob- lem when patterns are tested. Furthermore, it is easy to see that these problems are exacerbated as patterns become larger and more real (obviously, the examples of Fig. 1.1 are far from having enough resolution even to display normal type- fonts). In fact, a “combinatorial explosion” takes place: this is normally taken to mean that one or more parameters produce fast-varying (often exponential) effects, which “explode” as the parameters increase by modest amounts. Forgetting for the moment that the patterns of Fig. 1.1 have familiar shapes, let us temporarily regard them as random bit patterns. Now the number of bits in these N 3 N patterns is N2, and the number of possible patterns of this size is 2N2 : even in a case where N5 20, remembering all these patterns and their interpretations would be impossible on any practical machine, and searching systematically through them would take impracticably long (involving times of the order of the age of the universe). Thus it is not only impracticable to consider such brute force means of solving the recognition problem, but is also effectively impossible theoretically. These considerations show that other means are required to tackle the problem.

위에서 설명한 방식은 간단해 보이며 테스트 패턴의 왜곡이 발생하거나 노이즈가 존재하는 상황에도 대처할 수 있어 매우 효과적입니다.  
이는 테스트 패턴 (2) 및 (3)에 의해 설명됩니다.  
그러나 이러한 접근 방식이 항상 완벽한 것은 아닙니다.  
첫째, 왜곡이나 노이즈가 과도한 상황이 발생하여 해석 오류가 발생합니다.  
둘째, 패턴이 심하게 왜곡되거나 명백한 노이즈의 영향을 받지 않으면서도 잘못 해석되는 상황이 있습니다. 이는 노이즈나 왜곡의 합리적인 결과라기보다는 기술의 예상치 못한 한계를 나타내기 때문에 훨씬 더 심각한 문제로 보입니다.  
특히 테스트 패턴(6)과 마찬가지로 테스트 패턴이 적절한 학습 세트 패턴에 비해 변위되거나 방향이 잘못 지정되는 경우 이러한 문제가 발생합니다. 

13장, 기본 분류 개념에서 볼 수 있듯이, 위에서 언급한 예상치 못한 한계가 발생할 수 있는 강력한 원리가 있습니다:  
단순히 훈련 세트 패턴이 충분하지 않고, 존재하는 패턴이 실제 상황에서 발생할 수 있는 것을 충분히 대표하지 못한다는 것입니다.  
불행히도, 이는 큰 어려움을 초래합니다. 충분한 훈련 세트 패턴을 제공하면 심각한 저장 문제와 패턴을 테스트할 때 더욱 심각한 검색 문제가 발생하기 때문입니다.  
게다가, 이러한 문제들이 패턴이 점점 더 커지고 실제화됨에 따라 악화된다는 것을 쉽게 알 수 있습니다 (그림 1.1의 예시들은 일반적인 글꼴을 표시할 만큼 해상도가 충분하지 않다는 것을 알 수 있습니다).  
사실, "공생 폭발"이 일어납니다. 이는 보통 하나 이상의 매개변수가 빠르게 변하는 (종종 지수적인) 효과를 생성한다는 의미로 받아들여지며, 이는 매개변수가 약간 증가함에 따라 "폭발"됩니다.  
그림 1.1의 패턴이 익숙한 모양을 가지고 있다는 사실을 잠시 잊고, 일시적으로 무작위 비트 패턴으로 간주해 보겠습니다.  
이제 이 N*N 패턴에서 비트 수는 $N^2$이고, 이 크기의 가능한 패턴 수는 $2^{N^2}$입니다.  
N = 20 인 경우에도 이 모든 패턴과 그 해석을 기억하는 것은 어떤 실용적인 기계에서도 불가능하며, 체계적으로 검색하는 데는 우주의 나이 정도의 시간이 소요될 것입니다.  
따라서 인식 문제를 해결하기 위한 이러한 무차별 대입 수단을 고려하는 것은 비현실적일 뿐만 아니라 이론적으로도 사실상 불가능합니다.  
이러한 고려 사항은 문제를 해결하기 위해 다른 수단이 필요하다는 것을 보여줍니다.

### 1.2.2 TACKLING THE RECOGNITION PROBLEM
An obvious means of tackling the recognition problem is to standardize the images in some way. Clearly, normalizing the position and orientation of any 2D picture object would help considerably: indeed this would reduce the number of degrees of freedom by three. Methods for achieving this involve centralizing the objects—arranging that their centroids are at the center of the normalized image—and making their major axes (e.g., deduced by moment calculations) vertical or horizontal. Next, we can make use of the order that is known to be present in the image—and here it may be noted that very few patterns of real interest are indistinguishable from random dot patterns. This approach can be taken further: if patterns are to be nonrandom, isolated noise points may be eliminated. Ultimately, all these methods help by making the test pattern closer to a restricted set of training set patterns (although care must also be taken to process the train- ing set patterns initially so that they are representative of the processed test patterns). 
It is useful to consider character recognition further. Here we can make additional use of what is known about the structure of characters—namely, that they consist of limbs of roughly constant width. In that case the width carries no useful information, so the patterns can be thinned to stick figures (called skeletons—see Chapter 8: Binary Shape Analysis); then, hopefully, there is an even greater chance that the test patterns will be similar to appropriate training set patterns (Fig. 1.2). This process can be regarded as another instance of reducing the number of degrees of freedom in the image, and hence of helping to minimize the combinatorial explosion—or, from a practical point of view, to minimize the size of the training set necessary for effective recognition.

인식 문제를 해결하는 명백한 방법은 이미지를 어떤 방식으로든 표준화하는 것입니다.  
분명히, 2D 이미지 객체의 위치와 방향을 정규화하는 것이 상당히 도움이 될 것입니다.  
이는 실제로 자유도를 세 배로 줄일 수 있습니다.  
이를 달성하기 위한 방법으로는 객체를 중앙에 집중화하고, 객체의 중심이 정규화된 이미지의 중심에 위치하도록 배열하며, 주요 축(예: 모멘트 계산으로 추론)을 수직 또는 수평으로 만드는 것이 포함됩니다.  
다음으로, 이미지에 존재하는 것으로 알려진 순서를 활용할 수 있으며, 여기서 실제 관심 있는 패턴은 무작위 점 패턴과 구별할 수 없는 경우가 거의 없다는 점에 유의할 필요가 있습니다.  
이 접근 방식은 패턴이 무작위가 아닌 경우 고립된 노이즈 포인트를 제거할 수 있습니다.  
궁극적으로, 이러한 모든 방법은 테스트 패턴을 제한된 훈련 세트 패턴에 더 가깝게 만드는 데 도움이 됩니다(비록 훈련 세트 패턴이 처리된 테스트 패턴을 대표하도록 처음에 훈련 세트 패턴을 처리하는 데도 주의를 기울여야 함). 
문자 인식을 더 깊이 고려하는 것이 유용합니다.  
여기서 우리는 문자의 구조에 대해 알려진 것을 추가로 활용할 수 있습니다.  
즉, 대략 일정한 너비의 팔다리로 구성되어 있다는 점입니다.  
이 경우 너비는 유용한 정보를 담고 있지 않으므로 패턴을 막대 모양으로 얇게 만들 수 있습니다(골격이라고 함).  
그러면 테스트 패턴이 적절한 훈련 세트 패턴과 유사할 가능성이 훨씬 더 높아지기를 바랍니다(그림 1.2 참조).  

<img width="719" alt="스크린샷 2025-04-15 오전 12 24 19" src="https://github.com/user-attachments/assets/dd7df9b6-9bad-47d9-bee7-6d9ffe7bb4d7" />

이 과정은 이미지의 자유도를 줄이는 또 다른 예로 간주될 수 있으며, 따라서 조합 폭발을 최소화하거나, 실용적인 관점에서 효과적인 인식에 필요한 훈련 세트의 크기를 최소화하는 데 도움이 될 수 있습니다.

Next, consider a rather different way of looking at the problem. Recognition is necessarily a problem of discrimination—i.e., of discriminating between patterns of different classes. However, in practice, considering the natural variation of patterns, including the effects of noise and distortions (or even the effects of breakages or occlusions), there is also a problem of generalizing over patterns of the same class. In practical problems there is a tension between the need to discriminate and the need to generalize. Nor is this a fixed situation. Even for the character recognition task, some classes are so close to others (n’s and h’s will be similar) that less generalization is possible than in other cases. On the other hand, extreme forms of generalization arise when, for example, an A is to be recognized as an A whether it is a capital or small letter, or in italic, bold, suffix, or other form of font—even if it is handwritten. The variability is determined largely by the training set initially provided. What we emphasize here, however, is that generalization is as necessary a prerequisite to successful recognition as is discrimination.

다음으로, 문제를 바라보는 다소 다른 방식을 고려해 보세요.  
인식은 반드시 차별의 문제, 즉 서로 다른 클래스의 패턴을 구별하는 문제입니다.  
그러나 실제로는 노이즈와 왜곡(또는 파손이나 폐색의 영향)을 포함한 패턴의 자연스러운 변화를 고려할 때 동일 클래스의 패턴에 대한 일반화 문제도 발생합니다.  
실제 문제에서는 구별의 필요성과 일반화의 필요성 사이에 긴장감이 있습니다.  
또한 이것은 고정된 상황도 아닙니다. 문자 인식 작업의 경우에도 일부 클래스는 다른 클래스에 너무 가까워서(n과 h는 비슷할 것입니다) 다른 경우보다 일반화가 덜 가능합니다.  
반면에, 예를 들어 A가 대문자든 소문자든, 이탤릭체, 굵은 글씨, 접미사 또는 기타 형태의 글꼴이든 A로 인식되어야 하는 극단적인 형태의 일반화가 발생합니다.  
변동성은 처음에 제공된 교육 세트에 의해 크게 결정됩니다. 그러나 여기서 우리가 강조하는 것은 일반화가 차별인 만큼 성공적인 인식을 위한 필수 전제 조건으로 필요하다는 것입니다.

At this point it is worth considering more carefully the means whereby gener- alization was achieved in the examples cited above. First, objects were positioned and orientated appropriately; second, they were cleaned of noise spots; and third, they were thinned to skeleton figures (although the latter process is relevant only for certain tasks such as character recognition). In the last case, we are generaliz- ing over characters drawn with all possible limb widths, width being an irrelevant degree of freedom for this type of recognition task. Note that we could have generalized the characters further by normalizing their size and saving another degree of freedom. The common feature of all these processes is that they aim to give the characters a high level of standardization against known types of vari- ability before finally attempting to recognize them.

이 시점에서 위에서 언급한 예시들에서 일반화를 달성하는 방법을 더 신중하게 고려할 가치가 있습니다.  
첫째, 객체를 적절하게 배치하고 방향을 정했습니다. 둘째, 노이즈 스팟을 제거하고 셋째, 골격 도형으로 얇게 만들었습니다(비록 후자의 과정은 문자 인식과 같은 특정 작업에만 관련이 있지만).  
마지막 경우에는 모든 가능한 팔다리 너비로 그려진 문자에 대해 일반화하고 있으며, 너비는 이러한 유형의 인식 작업에 대해 무관한 자유도입니다.  
문자의 크기를 정규화하고 다른 자유도를 절약함으로써 문자를 더 일반화할 수 있었다는 점에 유의하세요.  
이 모든 과정의 공통점은 문자를 인식하기 전에 알려진 유형의 변동성에 대해 높은 수준의 표준화를 제공하는 것을 목표로 한다는 것입니다.

The standardization (or generalization) processes outlined above are all realized by image processing, i.e., the conversion of one image into another by suitable means. The result is a two-stage recognition scheme: first, images are converted into more amenable forms containing the same numbers of bits of data; and second, they are classified with the result that their data content is reduced to very few bits (Fig. 1.3). In fact, recognition is a process of data abstraction, the final data being abstract and totally unlike the original data. Thus we must imagine a letter A starting as an array of perhaps 20 3 20 bits arranged in the form of an A, and then ending as the 7 bits in an ASCII representation of an A, namely 1000001 (which is essentially a random bit pattern bearing no resemblance to an A). The last paragraph reflects to a large extent the history of image analysis. Early on, a good proportion of the image analysis problems being tackled were envisaged as consisting of an image “preprocessing” task carried out by image processing techniques, followed by a recognition task undertaken by pure pattern recognition methods (see Chapter 13: Basic Classification Concepts). These two topics—image processing and pattern recognition—consumed much research effort and effectively dominated the subject of image analysis, while “intermediate-level” approaches such as the Hough transform were, for a time, slower to develop. One of the aims of this book is to ensure that such intermediate-level processing techniques are given due emphasis, and indeed that the best range of techniques is applied to any computer vision task.

위에서 설명한 표준화(또는 일반화) 과정은 모두 이미지 처리, 즉 적절한 수단을 통해 한 이미지를 다른 이미지로 변환하는 것으로 실현됩니다.  
그 결과는 두 단계로 구성된 인식 체계입니다: 첫째, 이미지는 동일한 수의 데이터 비트를 포함하는 보다 적합한 형태로 변환되고, 둘째, 데이터 내용이 매우 적은 비트로 축소되는 결과로 분류됩니다(그림 1.3 참조).  

<img width="719" alt="스크린샷 2025-04-15 오전 12 29 14" src="https://github.com/user-attachments/assets/4b6d4daa-f2ad-48ee-a78a-96adb6b243b5" />

사실 인식은 데이터 추상화 과정으로, 최종 데이터는 추상적이고 원본 데이터와 완전히 다릅니다.  
따라서 문자 A가 A 형태로 배열된 약 20개의 320비트 배열로 시작하여 A의 ASCII 표현에서 7비트, 즉 1000001(본질적으로 A와 유사하지 않은 무작위 비트 패턴)로 끝나는 것을 상상해야 합니다.  
마지막 단락은 이미지 분석의 역사를 상당 부분 반영합니다.  
초기에는 이미지 분석 문제의 상당 부분이 이미지 처리 기술로 수행되는 이미지 "전처리" 작업과 순수 패턴 인식 방법으로 수행되는 인식 작업으로 구성될 것으로 예상되었습니다(제13장: 기본 분류 개념 참조).  
이 두 가지 주제인 이미지 처리와 패턴 인식은 많은 연구 노력을 필요로 하여 이미지 분석의 주제를 효과적으로 지배한 반면, Hough 변환과 같은 "중간 수준" 접근 방식은 한동안 개발 속도가 느렸습니다.  
이 책의 목표 중 하나는 이러한 중간 수준의 처리 기술이 충분히 강조되고 실제로 모든 컴퓨터 비전 작업에 최적의 기술 범위가 적용되도록 하는 것입니다.

### 1.2.3 OBJECT LOCATION
The problem that was tackled above—that of character recognition—is a highly constrained one. In a great many practical applications it is necessary to search pictures for objects of various types, rather than just interpreting a small area of a picture.

위에서 다룬 문제, 즉 문자 인식 문제는 매우 제한적인 문제입니다.  
많은 실용적인 응용 분야에서는 그림의 작은 영역을 해석하는 것이 아니라 다양한 유형의 물체를 검색하는 것이 필요합니다.

Search is a task that can involve prodigious amounts of computation and is also subject to a combinatorial explosion. Imagine the task of searching for a letter E in a page of text. An obvious way of achieving this is to move a suitable “template” of size n 3 n over the whole image, of size N 3 N, and to find where a match occurs (Fig. 1.4). A match can be defined as a position where there is exact agreement between the template and the local portion of the image but, in keeping with the ideas of Section 1.2.1, it will evidently be more relevant to look for a best local match (i.e., a position where the match is locally better than in adjacent regions) and where the match is also good in some more absolute sense, indicating that an E is present. One of the most natural ways of checking for a match is to measure the Hamming distance between the template and the local n 3 n region of the image, i.e., to sum the number of differences between corresponding bits. This is essentially the process described in Section 1.2.1. Then places with a low Hamming distance are places where the match is good. These template-matching ideas can be extended to cases where the corresponding bit positions in the template and the image do not just have binary values but may have intensity values over a range 0 255. In that case the sums obtained are no longer Hamming distances but may be generalized to the form:

검색은 엄청난 양의 계산을 필요로 할 수 있는 작업이며 조합론적 폭발의 영향을 받기도 합니다.  
텍스트 페이지에서 문자 E를 검색하는 작업을 상상해 보세요.  
이를 달성하는 명확한 방법은 크기 n * n인 적절한 "템플릿"을 전체 이미지 위에 이동시키고, 크기 N * N인 적절한 "템플릿"을 이동하여 일치하는 위치를 찾는 것입니다(그림 1.4).  
일치는 템플릿과 이미지의 로컬 부분 사이에 정확한 일치가 있는 위치로 정의할 수 있지만, 섹션 1.2.1의 아이디어에 따르면 최적의 로컬 일치(즉, 일치하는 위치가 인접한 영역보다 로컬에서 더 나은 위치)와 일치하는 위치를 찾는 것이 더 관련이 있을 것이며, 이는 E가 존재함을 나타냅니다.  
일치 여부를 확인하는 가장 자연스러운 방법 중 하나는 템플릿과 이미지의 로컬 n*n인 영역 사이의 해밍 거리를 측정하는 것, 즉 해당 비트 간의 차이 수를 합산하는 것입니다.  
이는 본질적으로 섹션 1.2.1에서 설명한 프로세스입니다. 그런 다음 해밍 거리가 낮은 장소가 일치하는 장소입니다.  
이러한 템플릿 일치 아이디어는 템플릿과 이미지의 해당 비트 위치가 이진 값뿐만 아니라 0~255 범위에서 강도 값을 가질 수 있는 경우로 확장할 수 있습니다.  
이 경우 얻은 합계는 더 이상 해밍 거리가 아니라 형식으로 일반화할 수 있습니다:

<img width="425" alt="스크린샷 2025-04-15 오전 12 33 32" src="https://github.com/user-attachments/assets/8036b38a-750d-4bc8-942d-eecdd8cf1375" />

It being the local template value, Ii being the local image value, and the sum being taken over the area of the template. This makes template matching practica- ble in many situations: the possibilities are examined in more detail in subsequent chapters.

I_t 는 로컬 템플릿 값, I_i 는 로컬 이미지 값, 그리고 템플릿의 영역을 차지하는 합계입니다.  
이로 인해 템플릿 매칭은 많은 상황에서 실행 가능합니다. 가능성은 다음 장에서 더 자세히 살펴봅니다.

We referred above to a combinatorial explosion in this search problem too. The reason this arises is as follows. First, when a 5 3 5 template is moved over an N 3 N image in order to look for a match, the number of operations required is of the order of 52N2, totaling some 1 million operations for a 256 3 256 image. The problem is that when larger objects are being sought in an image, the number of operations increases as the square of the size of the object, the total number of operations being N2 n 2 when an n 3 n template is used. For a 30 3 30 template and a 256 3 256 image, the number of operations required rises toB60 million. Note that, in general, a template will be larger than the object it is used to search for, because some background will have to be included to help demarcate the object. Next, recall that in general, objects may appear in many orientations in an image (E’s on a printed page are exceptional). If we imagine a possible 360 orientations (i.e., one per degree of rotation), then a corresponding number of templates will in principle have to be applied in order to locate the object. This additional degree of freedom pushes the search effort and time to enormous levels, so far away from the possibility of real-time implementation that new approaches must be found for tackling the task. [“Real-time” is a commonly used phrase meaning that the information has to be processed as it becomes available: this contrasts with the many situations (such as the processing of images from space probes) where the information may be stored and processed at leisure.] Fortunately, many researchers have applied their minds to this problem and there are a many good ideas for tackling it. Perhaps the most important general means for saving effort on this sort of scale is that of two-stage (or multistage) template matching. The principle is to search for objects via their features. For example, we might consider searching for E’s by looking for characters that have horizontal line segments within them. Similarly, we might search for hinges on a manufacturer’s conveyor by looking first for the screw holes they possess. In general it is useful to look for small features, since they require smaller templates and hence involve significantly less computation, as demonstrated above. This means that it may be better to search for E’s by looking for corners instead of horizontal line segments.

이 검색 문제에서도 조합론적 폭발을 언급했습니다. 이러한 현상이 발생하는 이유는 다음과 같습니다.  
첫째, 일치하는 항목을 찾기 위해 5 * 5 템플릿을 N * N 이미지 위로 이동시키면 필요한 연산 수는 $5^2N^2$ 정도로, 256 * 256 이미지에 대해 총 100만 개의 연산이 발생합니다.  
문제는 이미지에서 더 큰 객체를 찾을 때 객체 크기의 제곱에 따라 연산 수가 증가하고, n * n 템플릿을 사용할 때 총 연산 수는 $N^2n^2$ 가 된다는 것입니다.  
30 * 30 템플릿과 256 * 256 이미지의 경우 필요한 연산 수는 6천만 개로 증가합니다.  
일반적으로 템플릿은 객체를 구분하는 데 도움이 되는 배경이 포함되어야 하기 때문에 검색하는 데 사용되는 객체보다 더 클 수 있습니다.  
다음으로, 일반적으로 객체는 이미지에서 여러 방향으로 나타날 수 있다는 점을 기억하세요(인쇄된 페이지의 E는 예외적입니다).  
가능한 360도 방향(즉, 회전 정도당 하나)을 상상한다면, 객체를 찾기 위해 원칙적으로 해당 수의 템플릿을 적용해야 합니다.  
이러한 추가 자유도는 검색 노력과 시간을 엄청난 수준으로 끌어올리며, 작업을 해결하기 위해 새로운 접근 방식을 찾아야 하는 실시간 구현의 가능성과는 거리가 멉니다.  
["실시간"]은 정보가 이용 가능해짐에 따라 처리되어야 한다는 의미로, 이는 정보를 저장하고 처리할 수 있는 많은 상황(예: 우주 탐사선에서 이미지를 처리하는 것)과 대조됩니다.  
다행히도 많은 연구자들이 이 문제에 마음을 쏟았고 이를 해결하기 위한 좋은 아이디어도 많이 있습니다.  
이러한 규모에서 노력을 절약하는 가장 중요한 일반적인 방법은 2단계(또는 다단계) 템플릿 매칭입니다.  
원칙은 객체의 특징을 통해 검색하는 것입니다. 예를 들어, 그 안에 수평선 세그먼트가 있는 문자를 찾아 E를 검색하는 것을 고려할 수 있습니다.  
마찬가지로 제조업체의 컨베이어에 있는 나사 구멍을 먼저 찾아 경첩을 검색할 수도 있습니다.  
일반적으로 위에서 설명한 것처럼 작은 템플릿이 필요하기 때문에 계산량이 훨씬 적기 때문에 작은 특징을 찾는 것이 유용합니다.  
즉, 수평 선분 대신 모서리를 찾아 E를 검색하는 것이 더 나을 수 있습니다.

Unfortunately, noise and distortions give rise to problems if we search for objects via small features—there is a risk of missing the object altogether. Hence it is necessary to collate the information from a number of such features. This is the point where the many available methods start to differ from each other. How many features should be collated? Is it better to take a few larger features than many smaller ones? And so on. Also, we have not answered in full the question of what types of feature are the best to employ. These and other questions are considered in the subsequent chapters. Indeed, in a sense, these questions are the subject of this book. Search is one of the fundamental problems of vision, yet the details and the application of the basic idea of two-stage template matching give the subject much of its richness: to solve the recognition problem, the data set needs to be explored carefully. Clearly, any answers will tend to be data-dependent but it is worth exploring to what extent there are generalized solutions to the problem.

안타깝게도 작은 특징을 통해 물체를 검색하면 노이즈와 왜곡으로 인해 문제가 발생하여 물체를 완전히 놓칠 위험이 있습니다.  
따라서 이러한 여러 특징에서 정보를 수집해야 합니다. 바로 이 지점에서 사용 가능한 여러 가지 방법이 서로 다르기 시작합니다.  
몇 가지 특징을 수집해야 할까요? 작은 특징보다 큰 특징을 몇 가지 취하는 것이 더 나을까요? 등의 질문이 있습니다.  
또한 어떤 유형의 특징을 사용하는 것이 가장 좋은지에 대한 질문에 대해 완전히 답변하지 않았습니다.  
이러한 질문과 다른 질문은 다음 장에서 고려합니다. 실제로 어떤 의미에서 이러한 질문은 이 책의 주제입니다.  
검색은 시각의 근본적인 문제 중 하나이지만, 2단계 템플릿 매칭이라는 기본 아이디어의 세부 사항과 적용은 인식 문제를 해결하기 위해 데이터 세트를 신중하게 탐색해야 한다는 점에서 많은 풍부함을 제공합니다.  
분명히 모든 답변은 데이터에 의존하는 경향이 있지만 문제에 대한 일반화된 해결책이 어느 정도 있는지 살펴볼 가치가 있습니다.

### 1.2.4 SCENE ANALYSIS
The last subsection considered what is involved in searching an image for objects of a certain type: the result of such a search is likely to be a list of centroid coordinates for these objects, although an accompanying list of orientations might also be obtained. This subsection considers what is involved in scene analysis—the activity we are continually engaged in as we walk around, negotiating obstacles, finding food, and so on. Scenes contain a multitude of objects, and it is their interrelationships and relative positions that matter as much as identifying what they are. It may seem that there is no need for a search per se and that we could passively take in what is in the scene. However, there is much evidence (e.g., from analysis of eye movements) that the eye brain system interprets scenes by continually asking questions about what is there. For example, we might ask the following questions: Is this a lamppost? How far away is it? Do I know this person? Is it safe to cross the road? And so on. It is not the purpose here to dwell on these human activities or introspection about them but merely to observe that scene analysis involves enormous amounts of input data, complex relationships between objects within scenes and, ultimately, descriptions of these complex relationships. The latter no longer take the form of simple classification labels, or lists of object coordinates, but have a much richer information content: indeed, a scene will, to a first approximation, be better described in English than as a list of numbers. It seems likely that a much greater combinatorial explosion is involved in determining relationships between objects than in merely identifying and locating them. Hence, all sorts of props must be used to aid visual interpretation: there is considerable evidence of this in the human visual system, where contextual information and the availability of immense databases of possibilities clearly help the eye to a considerable degree. Note also that scene descriptions may initially be at the level of factual content but will eventually be at a deeper level—that of meaning, significance, and relevance. However, we shall not be able to delve further into these areas in this book.

마지막 하위 섹션에서는 특정 유형의 물체를 검색할 때 무엇이 관련되어 있는지 고려했습니다:  
이러한 검색의 결과는 이러한 물체에 대한 중심 좌표 목록일 가능성이 높지만, 함께 제공되는 방향 목록도 얻을 수 있습니다.  
이 하위 섹션에서는 장면 분석에 관련된 것, 즉 걸어 다니면서 지속적으로 참여하는 활동, 장애물 협상, 음식 찾기 등을 고려합니다.  
장면에는 여러 물체가 포함되어 있으며, 그들의 상호 관계와 상대적 위치가 무엇인지 식별하는 것만큼이나 중요합니다.  
검색 자체가 필요 없어 보일 수 있으며 장면에 있는 것을 수동적으로 받아들일 수 있습니다.  
그러나 눈 움직임 분석을 통해 눈의 뇌 시스템이 무엇이 있는지 지속적으로 질문함으로써 장면을 해석한다는 많은 증거(예: 눈의 움직임 분석)가 있습니다.  
예를 들어, 우리는 다음과 같은 질문을 할 수 있습니다: 이것이 가로등인가요? 얼마나 멀리 떨어져 있나요? 이 사람을 알고 있나요? 길을 건너는 것이 안전한가요? 등입니다.  
여기서의 목적은 이러한 인간 활동이나 성찰에 머무르는 것이 아니라 장면 분석이 방대한 양의 입력 데이터, 장면 내 물체 간의 복잡한 관계, 궁극적으로 이러한 복잡한 관계에 대한 설명을 포함한다는 것을 관찰하는 것입니다.  
후자는 더 이상 단순한 분류 레이블이나 물체 좌표 목록의 형태를 취하지 않고 훨씬 더 풍부한 정보 내용을 가지고 있습니다.  
실제로 장면은 처음에는 숫자 목록보다 영어로 더 잘 설명될 것입니다.  
단순히 물체를 식별하고 찾는 것보다 훨씬 더 큰 조합적 폭발이 물체 간의 관계를 결정하는 데 관여하는 것 같습니다.  
따라서 시각적 해석을 돕기 위해 모든 종류의 소품을 사용해야 합니다: 인간 시각 시스템에는 맥락 정보와 방대한 가능성 데이터베이스의 가용성이 눈에 상당한 도움이 되는 상당한 증거가 있습니다.  
또한 장면 묘사는 처음에는 사실적인 내용 수준에 불과할 수 있지만 결국에는 의미, 중요성, 관련성 등 더 깊은 수준에 도달할 것입니다.  
그러나 이 책에서는 이러한 영역에 대해 더 자세히 설명할 수 없습니다.

### 1.2.5 VISION AS INVERSE GRAPHICS
It has often been said that vision is “merely” inverse graphics. There is a certain amount of truth in this. Computer graphics is the generation of images by computer, starting from abstract descriptions of scenes and knowledge of the laws of image formation. Also, it is difficult to quarrel with the idea that vision is the process of obtaining descriptions of sets of objects, starting from sets of images and a knowledge of the laws of image formation (indeed, it is good to see a definition that explicitly brings in the need to know the laws of image formation, since it is all too easy to forget that this is a prerequisite when building descriptions incorporating heuristics that aid interpretation). However, this similarity in formulation of the two processes hides some fundamental points. First, graphics is a “feedforward” activity, i.e., images can be produced straightforwardly once sufficient specification about the viewpoint and the objects, and knowledge of the laws of image formation, have been obtained. True, considerable computation may be required but the process is entirely determined and predictable. The situation is not so straightforward for vision because search is involved and there is an accompanying combinatorial explosion. Indeed, some vision packages incorporate graphics (or CAD) packages (Tabandeh and Fallside, 1986), which are inserted into feedback loops for interpretation: the graphics package is then guided iteratively until it produces an acceptable approximation to the input image, when its input parameters embody the correct interpretation (there is a close parallel here with the problem of designing analog-to-digital converters by making use of digital-to-analog converters). Hence, it seems inescapable that vision is intrinsically more complex than graphics. We can clarify the situation somewhat by noting that, as a scene is observed, a 3-D environment is compressed into a 2-D image and a considerable amount of depth and other information is lost. This can lead to ambiguity of interpretation of the image (both a helix viewed end-on and a circle project into a circle), so the 3-D to 2-D transformation is many-to-one. Conversely, the interpretation must be one-to-many, meaning that there are many possible interpretations, yet we know that only one can be correct: vision involves not merely providing a list of all possible interpretations but providing the most likely one. Hence, some additional rules or constraints must be involved in order to determine the single most likely interpretation. Graphics, in contrast, does not have these problems, as the above ideas show it to be a many-to-one process.

비전은 종종 "단순히" 역그래픽이라고 말해왔습니다. 여기에는 어느 정도의 진실이 담겨 있습니다.  
컴퓨터 그래픽은 장면의 추상적인 묘사와 이미지 형성 법칙에 대한 지식에서 시작하여 컴퓨터가 이미지를 생성하는 과정입니다.  
또한, 비전이 이미지 세트와 이미지 형성 법칙에 대한 지식에서 시작하여 객체 세트의 설명을 얻는 과정이라는 생각에 이의를 제기하기 어렵습니다(사실, 이미지 형성 법칙을 알아야 할 필요성을 명시적으로 가져오는 정의를 보는 것이 좋습니다. 왜냐하면 해석을 돕는 휴리스틱을 포함한 설명을 구축할 때 이것이 전제 조건이라는 것을 잊기가 너무 쉽기 때문입니다).  
그러나 이러한 두 프로세스의 형식적 유사성은 몇 가지 근본적인 점을 숨기고 있습니다.  
첫째, 그래픽은 "피드포워드" 활동으로, 즉 시점과 객체에 대한 충분한 사양이 확보되고 이미지 형성 법칙에 대한 지식이 확보되면 이미지를 간단하게 생성할 수 있습니다.  
사실 상당한 계산이 필요할 수 있지만 프로세스는 전적으로 결정되고 예측 가능합니다.  
검색이 필요하고 조합적 폭발이 수반되기 때문에 비전에 상황은 그렇게 간단하지 않습니다.  
실제로 일부 비전 패키지에는 그래픽(또는 CAD) 패키지(Tabande와 Fallside, 1986)가 포함되어 있으며, 이 패키지는 해석을 위해 피드백 루프에 삽입됩니다:  
그래픽 패키지는 입력 매개변수가 올바른 해석을 구현할 때까지 반복적으로 안내되며, 입력 매개변수가 입력 이미지에 대한 허용 가능한 근사치를 생성합니다(디지털-아날로그 변환기를 사용하여 아날로그-디지털 변환기를 설계하는 문제와 밀접한 유사점이 있습니다).  
따라서 비전이 그래픽보다 본질적으로 더 복잡하다는 것은 피할 수 없는 일처럼 보입니다.  
우리는 장면이 관찰되면 3D 환경이 2D 이미지로 압축되고 상당한 깊이와 기타 정보가 손실된다는 점을 언급함으로써 상황을 어느 정도 명확히 할 수 있습니다.  
이로 인해 이미지 해석이 모호해질 수 있으므로(엔드온으로 보는 나선과 원을 원으로 투영하는 것 모두) 3D에서 2D로의 변환은 다대일로 이루어집니다.  
반대로 해석은 일대일로 이루어져야 하므로 가능한 해석이 많지만, 우리는 한 가지만 정확할 수 있다는 것을 알고 있습니다: 비전은 단순히 모든 가능한 해석의 목록을 제공하는 것뿐만 아니라 가장 가능성이 높은 해석을 제공하는 것을 포함합니다.  
따라서 가장 가능성이 높은 단일 해석을 결정하기 위해서는 몇 가지 추가 규칙이나 제약이 필요합니다.  
반면 그래픽은 위의 아이디어에서 알 수 있듯이 이러한 문제가 없습니다.

## 1.3 FROM AUTOMATED VISUAL INSPECTION TO SURVEILLANCE
So far we have considered the nature of vision but not what man-made vision systems may be used for. There is in fact a great variety of applications for artificial vision systems—including, of course, all of those for which man employs his visual senses. Of particular interest in this book are surveillance, automated inspection, robot assembly, vehicle guidance, traffic monitoring and control, biometric measurement, and analysis of remotely sensed images. By way of example, fingerprint analysis and recognition have long been important applications of computer vision, as have the counting of red blood cells, signature verification and character recognition, and aeroplane identification (both from aerial silhouettes and from ground surveillance pictures taken from satellites). Face recognition and even iris recognition have become practical possibilities and vehicle guidance by vision will in principle soon be sufficiently reliable for urban use. Whether the public will accept this, with all its legal implications, is another matter, but note that radar blind-landing aids for aircraft have been in wide use for some years. In fact, last-minute automatic action to prevent accidents is a good compromise (see Chapter 24: Epilogue—Perspectives in Vision, for a related discussion on driver assistance schemes). Among the applications of vision considered in this book are those of manufacturing industry—particularly, automated visual inspection and vision for automated assembly. In these cases, much of the same manufactured components are viewed by cameras: the difference lies in how the resulting information is used. In assembly, components must be located and orientated so that a robot can pick them up and assemble them. For example, the various parts of a motor or brake system need to be taken in turn and put into the correct positions, or a coil may have to be mounted on a television tube, an integrated circuit placed on a printed circuit board, or a chocolate placed into a box. In inspection, objects may pass the inspection station on a moving conveyor at rates typically between 10 and 30 items per second, and it has to be ascertained whether they have any defects. If any defects are detected, the offending parts will usually have to be rejected: that is the feedforward solution. In addition, a feedback solution may be instigated—i.e., some parameter may have to be adjusted to control plant further back down the production line (this is especially true for parameters that control dimensional characteristics such as product diameter). Inspection also has the potential for amassing a wealth of information that is useful for management, on the state of the parts coming down the line: the total number of products per day, the number of defective products per day, the distribution of sizes of products, and so on. 

지금까지 우리는 시각의 본질에 대해 고려해왔지만, 인공 시각 시스템이 어떤 용도로 사용될 수 있는지는 고려하지 않았습니다.  
사실 인공 시각 시스템에는 인간이 시각적 감각을 활용하는 모든 시스템을 포함하여 매우 다양한 응용 분야가 있습니다.  
이 책에서 특히 흥미로운 것은 감시, 자동 검사, 로봇 조립, 차량 안내, 교통 모니터링 및 제어, 생체 측정 및 원격 감지 이미지 분석입니다.  
예를 들어, 지문 분석과 인식은 적혈구 수, 서명 확인 및 문자 인식, 항공기 식별(항공 실루엣과 위성에서 촬영한 지상 감시 사진 모두)과 같은 컴퓨터 시각의 중요한 응용 분야로 오랫동안 사용되어 왔습니다.  
얼굴 인식과 홍채 인식은 실용적인 가능성이 되었으며, 시각에 의한 차량 안내는 원칙적으로 곧 도시에서 사용하기에 충분히 신뢰할 수 있게 될 것입니다.  
대중이 이를 받아들일지 여부는 다른 문제이지만, 항공기의 레이더 블라인드 착륙 보조 장치는 몇 년 동안 널리 사용되어 왔습니다.  
사실 사고를 예방하기 위한 막바지 자동 조치는 좋은 타협안입니다(제24장: 에필로그—시각의 관점, 운전자 지원 제도에 대한 관련 논의 참조).  
이 책에서 고려하는 시각의 응용 분야 중에는 제조업, 특히 자동 시각 검사와 시각이 포함됩니다.  
이러한 경우 동일한 제조 부품의 대부분을 카메라로 볼 수 있습니다: 차이점은 결과 정보가 어떻게 사용되는지에 있습니다.  
조립 시 로봇이 부품을 집어 들고 조립할 수 있도록 부품을 위치하고 배치해야 합니다.  
예를 들어, 모터나 브레이크 시스템의 다양한 부품을 차례로 올바른 위치에 배치하거나, 텔레비전 튜브, 인쇄 회로 기판에 배치된 집적 회로 또는 상자에 담긴 초콜릿에 코일을 장착해야 할 수도 있습니다.  
검사 시 물체는 일반적으로 초당 10~30개의 품목을 이동하는 컨베이어를 통해 검사 스테이션을 통과할 수 있으며, 결함이 있는지 여부를 확인해야 합니다.  
결함이 감지되면 일반적으로 문제가 되는 부품은 거부되어야 하는데, 이는 피드포워드 솔루션입니다.  
또한 피드백 솔루션을 유도할 수 있으며, 즉 생산 라인을 따라 공장을 제어하기 위해 일부 매개변수를 조정해야 할 수도 있습니다(제품 직경과 같은 치수 특성을 제어하는 매개변수의 경우 특히 그렇습니다).  
검사는 또한 하루 총 제품 수, 하루 불량 제품 수, 제품 크기 분포 등 관리에 유용한 풍부한 정보를 수집할 수 있는 잠재력을 가지고 있습니다.

An important feature of most industrial tasks is that they take place in real time: if it is used, vision must be able to keep up with the manufacturing process. For assembly, this may not be too exacting a problem, since a robot may not be able to pick up and place more than one item per second—leaving the vision system a similar time to do its processing. For inspection, this supposition is rarely valid: even a single automated line (e.g., one for stoppering bottles) is able to keep up a rate of 10 items per second (and, of course, parallel lines are able to keep up much higher rates). Hence, visual inspection tends to press computer hardware very hard, so care needs to be taken in the design of hardware accelerators for such applications. Finally, we return to the starting discussion about the huge variety of applications of vision, and it is interesting to consider surveillance tasks as the outdoor analogs of automated inspection (indeed, it is amusing to imagine that cars speeding along a road are just as subject to inspection as products speeding along a product line!). In fact, they have recently been acquiring close to exponentially increasing application. Thus the techniques used for inspection have acquired an injection of vitality, and many more techniques have been developed. Naturally, this has meant the introduction of whole tranches of new subject matter, such as motion analysis and perspective invariants (see Part 4, 3-D Vision and Motion). It is also interesting that such techniques add richness to topics as face recognition (see Chapter 21: Face Detection and Recognition: the Impact of Deep Learning).

대부분의 산업 작업의 중요한 특징 중 하나는 실시간으로 이루어진다는 점입니다: 비전을 사용할 경우, 비전은 제조 공정을 따라잡을 수 있어야 합니다.  
조립의 경우 로봇이 초당 하나 이상의 품목을 픽업하고 배치할 수 없기 때문에 비전 시스템이 처리하는 데 비슷한 시간이 걸릴 수 있기 때문에 이는 그리 까다로운 문제가 아닐 수 있습니다.  
검사의 경우, 이러한 가정은 거의 유효하지 않습니다: 단일 자동화된 라인(예: 병을 막는 라인)도 초당 10개의 품목을 유지할 수 있으며(물론 병렬 라인도 훨씬 높은 비율을 유지할 수 있습니다).  
따라서 시각적 검사는 컴퓨터 하드웨어를 매우 강하게 누르는 경향이 있으므로 이러한 애플리케이션을 위한 하드웨어 가속기 설계에 주의가 필요합니다.  
마지막으로, 비전의 다양한 응용 분야에 대한 논의로 돌아가, 자동화된 검사의 야외 아날로그로서 감시 작업을 고려하는 것은 흥미롭습니다(실제로 도로를 질주하는 자동차도 제품 라인을 따라 질주하는 것과 마찬가지로 검사 대상이라는 것을 상상해 보세요!).  
실제로 최근 기하급수적으로 증가하는 응용 분야에 근접하게 발전하고 있습니다. 따라서 검사에 사용되는 기술은 활력을 불어넣는 주입을 획득했으며, 더 많은 기술이 개발되었습니다.  
이는 자연스럽게 동작 분석 및 원근 불변량과 같은 새로운 주제의 전체 트랜치를 도입하는 것을 의미합니다(4부, 3-D 비전 및 동작 참조).  
이러한 기술이 얼굴 인식과 같은 주제에 풍부함을 더한다는 점도 흥미롭습니다(21장: 얼굴 감지 및 인식: 딥 러닝의 영향 참조).

## 1.4 WHAT THIS BOOK IS ABOUT
The foregoing sections have examined something of the nature of computer vision and have briefly considered its applications and implementation. It is already clear that implementing computer vision involves considerable practical difficulties but, more important, these practical difficulties embody substantial fundamental problems: these include various factors giving rise to excessive processing load and time. Practical problems may be overcome by ingenuity and care: however, by definition, truly fundamental limitations cannot be overcome by any means—the best that we can hope for is that we will be able to minimize their effects following a complete understanding of their nature. Understanding is thus a cornerstone for success in computer vision. It is often difficult to achieve, since the data set (i.e., all pictures that could reasonably be expected to arise) is highly variegated. Indeed, much investigation is required to determine the nature of a given data set, including not only the objects being observed but also the noise levels, degrees of occlusion, breakage, defect, and distortion that are to be expected, and the quality and nature of the lighting. Ultimately, sufficient knowledge might be obtained in a useful set of cases so that a good understanding of the milieu can be attained. Then it remains to compare and contrast the various methods of image analysis that are available. Some methods will turn out to be quite unsatisfactory for reasons of robustness, accuracy or cost of implementation, or other relevant variables: and who is to say in advance what a relevant set of variables is? This, too, needs to be ascertained and defined. Finally, among the methods that could reasonably be used, there will be competition: tradeoffs between parameters such as accuracy, speed, robustness, and cost will have to be worked out first theoretically and then in numerical detail to find an optimal solution. This is a complex and long process in a situation where workers have in the past aimed to find solutions for their own particular (often short-term) needs. Clearly, there is a need to ensure that practical computer vision advances from an art to a science. Fortunately this process has been developing for some years, and one of the aims of this book is to throw additional light on the problem.

앞서 언급한 섹션들은 컴퓨터 비전의 본질에 대해 검토했으며, 그 응용과 구현에 대해 간략히 살펴보았습니다.  
컴퓨터 비전을 구현하는 데 상당한 실질적인 어려움이 수반된다는 것은 이미 분명하지만, 더 중요한 것은 이러한 실질적인 어려움이 상당한 근본적인 문제를 내포하고 있다는 점입니다.  
여기에는 과도한 처리 부하와 시간이 발생하는 다양한 요인이 포함됩니다. 실질적인 문제는 독창성과 신중함으로 극복될 수 있지만, 정의상 진정한 근본적인 한계는 어떤 수단으로도 극복할 수 없습니다.  
우리가 기대할 수 있는 최선의 방법은 그 본질을 완전히 이해한 후 그 영향을 최소화할 수 있는 것입니다.  
따라서 이해는 컴퓨터 비전의 성공을 위한 초석입니다. 데이터 세트(즉, 합리적으로 발생할 것으로 예상되는 모든 그림)가 매우 다양하기 때문에 달성하기 어려운 경우가 많습니다.  
실제로 관찰되는 물체뿐만 아니라 예상되는 노이즈 수준, 가려짐, 파손, 결함, 왜곡 정도, 조명의 품질과 특성 등 주어진 데이터 세트의 본질을 파악하기 위해서는 많은 조사가 필요합니다.  
궁극적으로 환경에 대한 좋은 이해를 얻을 수 있도록 유용한 사례 세트에서 충분한 지식을 얻을 수 있을 것입니다.  
그런 다음 사용 가능한 다양한 이미지 분석 방법을 비교하고 대조해야 합니다.  
일부 방법은 견고성, 정확성 또는 구현 비용 또는 기타 관련 변수의 이유로 인해 상당히 불만족스러울 수 있습니다: 그리고 관련 변수 세트가 무엇인지 미리 말해야 하는 방법도 있습니다. 이 역시 확인하고 정의해야 합니다.  
마지막으로 합리적으로 사용할 수 있는 방법 중에서 경쟁이 있을 것입니다: 정확성, 속도, 견고성, 비용과 같은 매개변수 간의 균형을 먼저 이론적으로 해결한 다음 최적의 해결책을 찾기 위해 수치적으로 세부적으로 해결해야 합니다.  
이는 과거에 근로자들이 자신의 특정(종종 단기적인) 요구에 대한 해결책을 찾는 것을 목표로 했던 상황에서 복잡하고 긴 과정입니다.  
물론 실용적인 컴퓨터 비전이 예술에서 과학으로 발전할 수 있도록 해야 할 필요가 있습니다.  
다행히도 이 과정은 몇 년 동안 발전해 왔으며, 이 책의 목적 중 하나는 이 문제에 대한 추가적인 조명을 제공하는 것입니다.

Before proceeding further, there are one or two more pieces to fit into the jigsaw. First, there is an important guiding principle: if the eye can do it, so can the machine. Thus, if an object is fairly well hidden in an image, yet the eye can see it and track it, then it should be possible to devise a vision algorithm that can do the same. Next, although we can expect to meet this challenge, should we set our sights even higher and aim to devise algorithms that can beat the eye? There seems no reason to suppose that the eye is the ultimate vision machine: it has been built through the vagaries of evolution, so it may be well adapted for finding berries or nuts, or for recognizing faces, but ill-suited for certain other tasks. One such task is that of measurement. The eye probably does not need to measure the sizes of objects, at a glance, to better than a few percent accuracy. However, it could be distinctly useful if the robot eye could achieve remote size measurement, at a glance, and with an accuracy of say 0.001%. Clearly, the robot eye could acquire capabilities superior to those of biological systems. Again, this book aims to point out such possibilities where they exist. Finally, it will be useful to clarify the terms Machine Vision and Computer Vision. In fact, these arose a good many years ago when the situation was quite different from what it is today. Over time, computer technology has advanced hugely and at the same time knowledge about the whole area of vision has been radically developed. In the early days, Computer Vision meant the study of the science of vision and the possible design of the software—and to a lesser extent with what goes into an integrated vision system, whereas Machine Vision meant the study not only of the software but also of the hardware environment and of the image acquisition techniques needed for real applications—so it was a much more engineering-orientated subject. At this point in time, computer technology has advanced so far that a sizeable proportion of real-world and real-time applications can be realized on unaided PCs. This and the many developments in knowledge in this area have led to significant convergence between the terms, with the result that they are often used more or less interchangeably, although in this book we aim to unify the subject under the name Computer Vision.

더 나아가기 전에 직소에 넣을 조각이 하나 또는 두 개 더 있습니다.  
먼저 중요한 안내 원칙이 있습니다: 눈이 할 수 있다면 기계도 할 수 있다는 것입니다.  
따라서 물체가 이미지에 상당히 잘 숨겨져 있지만 눈이 이를 보고 추적할 수 있다면 동일한 작업을 수행할 수 있는 비전 알고리즘을 고안할 수 있어야 합니다.  
다음으로, 이 도전에 직면할 것으로 예상할 수 있지만 목표를 더 높게 설정하고 눈을 이길 수 있는 알고리즘을 고안해야 할까요?  
눈이 궁극적인 비전 기계라고 가정할 이유는 없어 보입니다: 진화의 다양한 과정을 거쳐 만들어졌기 때문에 베리나 견과류를 찾거나 얼굴을 인식하는 데 잘 적응할 수 있지만 다른 작업에는 적합하지 않습니다.  
이러한 작업 중 하나는 측정입니다. 눈은 한 눈에 보기에는 물체의 크기를 몇 퍼센트 이상의 정확도로 측정할 필요가 없을 것입니다.  
그러나 로봇 눈이 원격 크기 측정, 한 눈에 보기에는 0.001%의 정확도로 달성할 수 있다면 분명히 유용할 수 있습니다.  
분명히 로봇 눈은 생물학적 시스템보다 우수한 능력을 습득할 수 있습니다. 이 책은 이러한 가능성을 다시 한 번 지적하는 것을 목표로 합니다.  
마지막으로 머신 비전과 컴퓨터 비전이라는 용어를 명확히 하는 데 유용할 것입니다. 사실 이러한 용어는 현재와 상황이 상당히 달랐던 수년 전에 등장했습니다.  
시간이 지남에 따라 컴퓨터 기술은 비전의 전체 영역에 대한 지식이 급격히 발전했습니다.  
초창기 컴퓨터 비전은 비전의 과학과 소프트웨어의 가능한 설계에 대한 연구를 의미했으며, 머신 비전은 소프트웨어뿐만 아니라 실제 응용에 필요한 하드웨어 환경 및 이미지 획득 기술에 대한 연구를 의미했기 때문에 훨씬 더 공학적인 주제였습니다.  
이 시점에서 컴퓨터 기술은 지금까지 발전하여 실제 애플리케이션과 실시간 애플리케이션의 상당 부분을 비보조 PC에서 구현할 수 있게 되었습니다.  
이와 이 분야의 많은 지식 발전은 용어 간에 상당한 융합을 가져왔으며, 그 결과 용어는 종종 다소 혼용되어 사용되지만, 이 책에서는 컴퓨터 비전이라는 이름으로 주제를 통합하는 것을 목표로 합니다.

## 1.5 THE PART PLAYED BY MACHINE LEARNING
During the whole period that computer vision was developing in the way described above, the subject of pattern recognition was also progressing. Basic ideas on pattern recognition that started with Bayes theory and the nearest neighbor approach gradually changed with the advent of artificial neural net- works, which were designed to emulate the neuron networks known to exist in the human brain. In addition, other methods such as support vector machines and boosting arrived on the scene: then, during the last decade or so, “deep learning” came into prominence. All these techniques led to a new subject called Machine Learning, which embodies pure pattern recognition but emphasizes not only minimization of error rates but also systematic inclusion of probability and mathematical optimization. The impact of this subject on computer vision has been increasingly dramatic over the past decade and partic- ularly during the past 4 5 years. This book aims to include this development as an integral part of its coverage: Chapters 2, Images and Imaging Operations and Chapter 13, Basic Classification Concepts, introduce in turn the imaging side of computer vision and the machine learning side, while Chapter 15, Deep Learning Networks, leads the reader into the newer area of deep learning.

컴퓨터 비전이 위에서 설명한 방식으로 발전하는 동안 패턴 인식의 주제도 발전하고 있었습니다.  
베이즈 이론과 가장 가까운 이웃 접근 방식에서 시작된 패턴 인식에 대한 기본 아이디어는 인간의 뇌에 존재하는 것으로 알려진 뉴런 네트워크를 모방하도록 설계된 인공 신경망의 등장과 함께 점차 변화했습니다.  
또한 서포트 벡터 머신과 부스팅과 같은 다른 방법들도 등장했는데, 그 후 지난 10여 년 동안 '딥 러닝'이 주목받기 시작했습니다.  
이러한 모든 기술은 순수 패턴 인식을 구현할 뿐만 아니라 오류율의 최소화뿐만 아니라 확률과 수학적 최적화의 체계적인 포함을 강조하는 머신 러닝이라는 새로운 주제로 이어졌습니다.  
이 주제가 컴퓨터 비전에 미치는 영향은 지난 10년 동안, 특히 지난 4년 동안 점점 더 극적으로 나타났습니다.  
이 책은 이러한 발전을 포괄적인 부분으로 포함하는 것을 목표로 합니다: 2장, 이미지 및 이미징 작업과 13장, 기본 분류 개념은 컴퓨터 비전의 이미징 측면과 머신 러닝 측면을 차례로 소개하고, 15장, 딥 러닝 네트워크는 독자를 새로운 딥 러닝 영역으로 안내합니다.

Chapters 2 and 13 form the introductions to the two main branches of the subject—image processing and machine learning. Chapters 2 7 follow the image-processing theme, covering low-level vision and various widely used segmentation techniques, ranging from thresholding, through edge and feature detection to texture analysis. Chapters 8 12 move on to intermediate-level processing, which has developed significantly in the past two decades and is important for the inference of complex objects: to this end, key model-based vision techniques such as the Hough transform and RANSAC are covered in detail (Chapters 10 and 11). Active shape models (Chapter 12) are also important for many practical applications. However, the latter require knowledge of PCA and other machine learning concepts: these are covered in Chapter 14. Chapters 16 19 develop the subject of 3-D vision, while Chapter 20 introduces motion. Chapters 21 23 attend to three key application areas—face detection and recognition; surveillance; and in-vehicle vision systems. Chapter 24 reiterates and highlights some of the lessons and topics dealt with in the book; Appendix A develops the subject of Robust Statistics, which relates to a large proportion of the methods that are covered here; and Appendix B covers a topic that is essential background to a subject such as vision—namely the sampling theorem. Appendix C discusses the representation of color, while Appendix D is relevant to machine learning and contains important material on sampling from distributions.

2장과 13장은 이미지 처리와 기계 학습이라는 두 가지 주요 분야에 대한 소개를 구성합니다.  
2장~7장은 이미지 처리 주제를 따르며, 저수준 비전과 다양한 널리 사용되는 세분화 기법을 다룹니다. 이는 임계값 설정, 엣지 및 특징 감지, 텍스처 분석에 이르기까지 다양합니다.  
8장~12절은 지난 20년 동안 크게 발전하여 복잡한 객체의 추론에 중요한 중간 수준 처리로 넘어갑니다. 이를 위해 Hough 변환과 RANSAC와 같은 주요 모델 기반 비전 기술을 자세히 다룹니다(10장과 11장).  
능동형 형태 모델(12장)도 많은 실용적인 응용 분야에서 중요합니다. 그러나 후자는 PCA 및 기타 기계 학습 개념에 대한 지식이 필요합니다: 이는 14장에서 다룹니다.  
16장~19장은 3D 비전에 대한 주제를 개발하고, 20장은 동작을 소개합니다.  
21~23장은 얼굴 감지 및 인식, 감시, 차량 내 비전 시스템 등 세 가지 주요 응용 분야에 대해 다룹니다.  
24장은 이 책에서 다룬 몇 가지 교훈과 주제를 반복하고 강조합니다;  
부록 A는 여기서 다루는 방법의 많은 부분과 관련된 강건 통계학 주제를 개발하고, 부록 B는 비전과 같은 주제의 필수적인 배경, 즉 샘플링 정리를 다룹니다. 부록 C는 색상 표현에 대해 논의하고, 부록 D는 기계 학습과 관련이 있으며 분포에서 샘플링하는 중요한 자료를 포함하고 있습니다.


## findObject
### 카메라에서 색상 두 개를 잡아서 그 사이의 중점을 잡고 첫 번째 색에서 두 번째 색으로 화살표를 가르키면서 물체의 방향 및 움직임을 감지하는 프로젝트입니다.

저는 두개의 색 ( 팀컬러와 id컬러 )를 입력받아 팀컬러에서 id 컬러쪽으로 나가는 화살표를 그려내고 팀컬러의 중심좌표를 알아내 그 좌표를 띄워주는 형식으로 코드를 작성하였습니다. 이 프로그램 사용방법은 카메라 앞 네모칸에 첫 번째 색을 인식시킨 뒤 스페이스바를 누르면 색이 저장이 되고, 2번째 색깔도 인식을 시킨 뒤 스페이스바로 저장을 시킵니다. 그러면 첫 번째로 인식된 색에서 두 번째로 인식된 색으로 화살표가 생성이 됩니다.
 


제가 생각한 알고리즘으로 코드를 짜기 위해 먼저 cv2와 numpy를 import 시킵니다.
 
색깔 인식에 필요한 color 변수 2개와 각 전역 변수들을 선언해줍니다. (전역변수에 대한 설명은 추후에 사용할 때 하겠습니다.)


 
Trackbar에서 사용될 변수 4개를 생성합니다. 변수는 각각 saturation을 조절하기 위한 변수 2개와 명도를 조절하기 위한 변수 2개 입니다. 
 
각각의 인식시킨 색깔의 범위를 제한합니다. 넘파이어레이를 통해서 색의 범위값을 담아둡니다. 색을 일정한 색으로 지정할 수도 있지만 트랙바를 통해서 색의 범위를 조절할 수 있게 합니다. 이 코드는 사용자에게 색을 입력받아서 
 

그 다음 새 창 두개를 띄웁니다. 하나는 result 라는 창이고, 하나는 color 라는 창입니다. Result 에서는 트랙바를 띄워주고, 감지되어진 색깔만을 띄워줍니다. 트랙바를 이용해서 명도와 채도를 조절해 내가 원하는 색만 인식될 수 있도록 잘 조정해줍니다.
그 밑 코드는 트랙바를 만들고, 변수의 값을 set 하는 코드입니다.
Cv.createTrackbar() 함수를 통해서 saturation_th1 의 값을 사용자로 하여금 조정할 수 있게 만들어줍니다.

 

Videocapture(0) 함수를 통해서 웹캠 영상을 불러옵니다. 파라미터 안에 0을 넣으면 웹캠이 호출됩니다.

  
반복문을 계속돌리면서 작업을 수행합니다.
Cap.read()를 통해 영상을 읽어오고,
cv.flip(img_color, 1) 을 통해 좌우 반전을 시킵니다.
If ret == false: continue그리고 만약 영상이 나오지 않는다면 반복문을 종료시킵니다.

Img_hsv = cv.cvtColor(img_color2, cv.COLOR_BGR2HSV)
사용자에게 받아온 Color 값을 hsv 로 변환시키는 역할을 합니다. 

가운데 사각형을 만들기 위해 영상 전체의 사이즈 /2 로 가운데에 네모상자를 생성할 변수 Width 와 height 을 만듭니다.

색이 지정이되면 알려주기 위한 Set_color 변수를 만듭니다. 
색이 지정이 안되어있다면 가운데 빨강색의 네모상자를 만듭니다.

 

cv.inRange() 함수를 사용하여, 아까 사용자가 입력했던 색의 최대~ 최소값에 들어가는 수를 전부 0, 나머지는 1로 만들어 흑백사진으로 만들어준다.
img_maskA에 첫번째 색, img_maskB에 두번째 색이 들어간다.
그 후 cv.morphologyEX() 함수를 통해서 노이즈를 없애주는 작업을 한 뒤, 아까 img_maskA, img_maskB에 넣어두었던 애들을 or 연산자로 합쳐줍니다. 
Img_result= cv.bitwise_and(img_color, img_color,mask=img_maskC)
그다음 원본 이미지에서 범위에 해당하는 영상만 획득할 수 있도록 and 연산자를 사용하여
Img_result에 담습니다.

 
 
그 후 connectedComponentswithStats() 함수를 통해서 라벨링하여 영역을 추출합니다.

중심점을 잡기위해 centerX,center = int(centroid[0]), int(centroid[1]
한 뒤 그 값을 전역변수 centerAX, AY에 옮깁니다. 
PointX 변수는 만들고 안썼습니다.

유사범위가 1500 보다 높으면 중심좌표에 원을 그리고 그 라벨링된 전체에 테두리를 칠합니다. 위 코드를 색깔 2개에 전부 적용시킨 뒤

 

cv.arrowedLine()
2가지 색 중 처음 선택한 색의 중심좌표에서 2번째 식별된 색쪽으로 화살표를 그리는 함수를 사용합니다.

 

중심좌표를 받아오기 위한 코드입니다.
위쪽에서 centerX, centerY 변수가 있었긴하지만 
Puttext() 함수를 통해서 중심좌표를 화면에 표시하기 위해 썼던이 자꾸 에러가 나서 cv.moments() 라는 함수를 통해서 중심좌표를 다시 구하고, 그 중심조표를 puttext()함수를 이용해서 화면에 중심좌표 값을 띄워줍니다.

 
 

Space버튼을 누르면 roi 함수안에 사각형안에 들어온 색을 저장시킵니다.  한 번더 누르면 같은 동작을 반복하고, set_color가 되었다는 것을 알려줍니다. 

마지막으로 영상재생이 끝나면 (esc 키를 통해 break 시키면) videocapture object를 release 하고 window를 닫습니다.

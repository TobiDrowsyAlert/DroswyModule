from tkinter import messagebox
class sleep_step_calc:
    #status code 정의
    C_BLINK=100
    C_BLIND=101
    C_YAWN=200
    C_DRIVER_AWAY=300
    C_DIRVER_AWARE_FAIL=301
    C_NOMAL=400

    #일반 변수
    __sleep_weight=0
    __last_yawn=0
    __last_sleep_weight=0
    __sleep_service_flag=False
    def event_sleep_step(self):
        if self.__sleep_weight==3:
            if not self.__sleep_service_flag:
                # 졸음 1단계 서비스
                # 예시 메시지 박스
                messagebox.showinfo(title="경고!", message="졸음 단계 1단계 달성")
                self.__sleep_service_flag=True
        elif self.__sleep_weight==5:
            if not self.__sleep_service_flag:
                # 졸음 2단계 서비스
                # 예시 메시지 박스
                messagebox.showinfo(title="경고!", message="졸음 단계 2단계 달성")
                self.__sleep_service_flag = True
        elif self.__sleep_weight==7:
            if not self.__sleep_service_flag:
                # 졸음 3단계 서비스
                # 예시 메시지 박스
                messagebox.showinfo(title="경고!", message="졸음 단계 3단계 달성")
                self.__sleep_service_flag = True
        #else:
            #예외 처리

    def raise_sleep_weight(self):
        #졸음 가중치 상승
        __sleep_service_flag = False
        self.__last_sleep_weight=self.__sleep_weight
        self.__sleep_weight=self.__sleep_weight+1

    def raise_sleep_step(self):
        #졸음 단계 상승
        if self.__sleep_weight<3:
            __sleep_service_flag = False
            self.__last_sleep_weight = self.__sleep_weight
            self.__sleep_weight=3
        elif self.__sleep_weight>3 and self.__sleep_weight<5:
            __sleep_service_flag = False
            self.__last_sleep_weight = self.__sleep_weight
            self.__sleep_weight=5
        elif self.__sleep_weight>5 and self.__sleep_weight<7:
            __sleep_service_flag = False
            self.__last_sleep_weight = self.__sleep_weight
            self.__sleep_weight=7

    def cancle_sleep_weight(self):
        #졸음 가중치 상승 취소
        self.__sleep_weight=self.__last_sleep_weight

    def blink_detection(self, blink, blink_THRESHOLD):
        if blink > blink_THRESHOLD:
            # 알림 수행
            messagebox.showinfo(title="경고!", message="졸음 가중치 상승!")
            self.raise_sleep_weight(self)
            return True

    def yawn_detection(self,yawn):
        if yawn > 1:
            if self.__last_yawn!=yawn:
                # 알림 수행
                messagebox.showinfo(title="경고!", message="졸음 가중치 상승!")
                self.__sleep_weight += 1
                self.__last_yawn=yawn
                return True





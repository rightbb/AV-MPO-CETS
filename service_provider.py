#服务供应商#
import numpy as np
from parameter import *
import time
class secondST():
    def __init__(self,stask_id,number):
        self.stask_id=stask_id
        self.time=TimeList[stask_id]
        self.left_time=None
        self.ac_time=0
        self.operate=False
        self.number=number

class firstFT():
    def __init__(self,ftask_id,number):
        self.ftask_id=ftask_id  
        self.number=number
        self.stask_id_list=SecondLayerTask[ftask_id]
        self.stask_list=[secondST(eve,number) for eve in self.stask_id_list]
        self.ac_time=0
        self.isnotfinish=len(self.stask_list)
        self.position=None
        self.real_time=None
        self.togo_site=None

    def pop(self):
        self.isnotfinish-=1


np.random.seed(9)
class order():
    def __init__(self,order_number):
        FTlist = []
        order_random = np.random.randint(12, size=18)  
        for i in range(len(order_random)):
            FTlist.append(firstFT(order_random[i]+1, str(order_number) + '_'+str(i)) )
        self.FTlist=FTlist
        self.ac_time=0
        self.position=None
        self.now_site=None
        self.time=0
        self.order_number=order_number
        self.have_done=False

class workshop():
    def __init__(self,equipment_ids_list,equipment_number):
        equipments=[]
        for i in range(equipment_number):
            equipments += equipment_ids_list
        self.equipments=equipments
        self.equipments_state=[0 for i in range(len(self.equipments))]
        self.jishu=0
        self.worktime=0



    def matching(self,stask_list:'st list'):
        if stask_list ==[]:
            return [],None,None,0

        else:

            min_time=1e2
            for i in range(len(stask_list)):
                if stask_list[i].operate == True:
                    if min_time >= stask_list[i].left_time:
                        min_time = stask_list[i].left_time

            for i in range(len(stask_list)):#匹配
                for j in range(len(self.equipments_state)):
                    if stask_list[i].stask_id==self.equipments[j] and self.equipments_state[j]==0 and stask_list[i].operate==False:
                        self.equipments_state[j] = stask_list[i]
                        stask_list[i].operate = True
                        stask_list[i].left_time=stask_list[i].time
                        if stask_list[i].left_time <=min_time:
                            min_time=stask_list[i].left_time

            for rf in self.equipments_state:
                self.worktime=0
                if rf !=0:
                    self.worktime+=1
                    break

            stask_number=[]
            stask_ac_time=[]
            for eve in stask_list:
                eve.ac_time += min_time
                if eve.operate == True:
                    eve.left_time -= min_time
                if eve.left_time ==0:
                    stask_number.append(eve.number)
                    stask_ac_time.append(eve.ac_time)
                    stask_list.remove(eve)

                    self.equipments_state=[0 if x==eve else x for x in self.equipments_state]

            return stask_list,stask_number,stask_ac_time,self.worktime

class BaseServiceProvider():

    def __init__(self, site_id, relative_position, equipment_ids_list,equipment_number):
        self.site_id = site_id
        self.absolute_position = np.array([10, 10]) + np.array(
            [400 * relative_position[0], 400 * relative_position[1]])
        self.equipment_ids_list=equipment_ids_list
        self.equipment_number=equipment_number
        self.wait_beserved_ST=[]
        self.workshop=workshop(equipment_ids_list,equipment_number)


    def recieveFT(self,firstFT):
        """"
        执行某一级任务，将一级任务投喂到车间待加工队列中
        """
        if firstFT is not None:
            for eve in firstFT:
                self.wait_beserved_ST += eve.stask_list

    def step(self):
        """
        有二级完成即停止
        :return:对应的1级完成了一个二级
        """
        self.wait_beserved_ST,stask_number,stask_ac_time,worktime=self.workshop.matching(self.wait_beserved_ST)

        return stask_number,stask_ac_time,worktime

class ServiceProviders():
    def __init__(self):
        self.position=providers_positions
        self.providers=[]
        for i in range(0,18):#0~17
            self.providers.append(BaseServiceProvider(i+1,self.position[i+1],equipment_list[i+1],factory_kind_number[i+1]))
        self.FTlist=[]
        self.worktimeall = 0



    def recieveFT_list(self,match_firstFT_list:'list19'):
        """
        供应商接受到某时刻一级子任务[[],[class ft],..,class ft]
        :return:
        """
        assert len(match_firstFT_list)==18
        for i in range(0,18):
            self.providers[i].recieveFT(match_firstFT_list[i])
            if match_firstFT_list[i] is not None:
                self.FTlist += match_firstFT_list[i]


    def step(self):

        """
        输入的是当前所有的一级任务列表 self.FTlist
        :return:
        遍历一级任务，每次有二级任务完成的，找到对应的pop出上面step中的
        firstFT.ac_time+=secondFT.ac_time
        firstFT.pop
        核查有无一级任务isnotfinish==0，有的话就重新规划
        """
        if self.FTlist==[]:

            # print('-本次step没有任务进来--并且-------have-now-work-done-----------')
            return None,self.worktimeall
        else:
            be_poped_FT=[]
            flag_rescheduling=False
            while flag_rescheduling == False:
                self.worktimeall=0
                for i in range(0,18):
                    stask_number, stask_ac_time,worktime = self.providers[i].step()
                    self.worktimeall+=worktime
                    if stask_number != None and stask_ac_time != None:
                        for eve in self.FTlist:
                            for j in range(len(stask_number)):
                                if eve.number == stask_number[j]:  #找到二级对应的1级任务
                                    if eve.ac_time < stask_ac_time[j]:
                                        eve.ac_time=stask_ac_time[j] #加上最大的二级的任务
                                    eve.pop()

                for eve in self.FTlist:
                    if eve.isnotfinish == 0:
                        flag_rescheduling=True
                        be_poped_FT.append(eve)
                        self.FTlist.remove(eve)

                if len(self.FTlist)==0:
                    # print('*be_poped_FT1', be_poped_FT)
                    break

            if be_poped_FT==[]:
                print('又空了')
                time.sleep(20)
            # print('*------------本次step任务为-------')
            return be_poped_FT ,self.worktimeall

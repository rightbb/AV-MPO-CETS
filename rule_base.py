import math
import random

rule_base=['Random','MinDistance_InOrder_dispart','MaxDistance_InOrder_dispart','MinLoadrate_Mindistance_InOrder','MinLoadrate_Mindistance_InOrder_part']
def count_path_on_road(initial_pos, end_pos, speed):
    if end_pos == None:
        return 0
    else:
        return math.sqrt((end_pos[0] - initial_pos[0]) ** 2 + (end_pos[1] - initial_pos[1]) ** 2) / speed

def FTask_provider(SecondLayerTask,equipment_list):
    cz = dict()
    for i in range(len(SecondLayerTask)):
        for j in range(len(equipment_list)):
            if set(equipment_list[j]) >  set(SecondLayerTask[i]) or set(equipment_list[j]) == set(SecondLayerTask[i]):
                if i not in cz:
                    cz[i] = [j]
                else:
                    cz[i].append(j)
    del cz[0]
def pos_in_lists(seq, elem):
    cd=[]
    for l,j in enumerate(seq):
        if elem == j:
            cd.append(l)
    return cd#返回最小的所在序号
def scheduling(ordernumber_to_schedul, orders, action,Sp,FTask_providers_match,providers_positions):

    ordernumber_to_schedul = list(set(ordernumber_to_schedul))
    if ordernumber_to_schedul == []:
        return {}
    dict1 = dict()

    if action=='Random':
        random.seed(6)
        for eve in orders:
            for number in ordernumber_to_schedul:
                if eve.order_number == number:
                    choosed = random.choice(FTask_providers_match[eve.FTlist[0].ftask_id])
                    dict1[number] = choosed  # 0-,1-18
        return dict1

    if action=='MinLoad_Mindistance_InOrder':#满足要求的里面，首个最小负载(二级),并且距离最近
        for eve in orders:
            for number in ordernumber_to_schedul:
                if eve.order_number == number:
                    if eve.position ==None:
                        choosed = random.choice(FTask_providers_match[eve.FTlist[0].ftask_id])
                        dict1[number] = choosed  # 0-,1-18
                    else:
                        minc=[]
                        for ys in FTask_providers_match[eve.FTlist[0].ftask_id]:
                            jis=0
                            for ii in Sp[ys-1].workshop.equipments_state:
                                if ii != 0:
                                    jis += 1
                            minc.append(jis)
                        # indes=minc.index(min(minc))
                        indes=pos_in_lists(minc, min(minc))

                        indes_final=indes[0]
                        minds=count_path_on_road(eve.position, providers_positions[FTask_providers_match[eve.FTlist[0].ftask_id][indes_final]], 1)

                        if len(indes)!=1:
                            for evewe in indes:
                                ys=FTask_providers_match[eve.FTlist[0].ftask_id][evewe]
                                disys=count_path_on_road(eve.position, providers_positions[ys], 1)
                                if minds>disys:
                                    minds=disys
                                    indes_final=evewe
                        dict1[number]=FTask_providers_match[eve.FTlist[0].ftask_id][indes_final]

        return dict1

    if action=='MinLoad_Maxdistance_InOrder':#满足要求的里面，首个最小负载(二级),并且距离最远
        for eve in orders:
            for number in ordernumber_to_schedul:
                if eve.order_number == number:
                    if eve.position ==None:
                        choosed = random.choice(FTask_providers_match[eve.FTlist[0].ftask_id])
                        dict1[number] = choosed  # 0-,1-18
                    else:
                        minc=[]
                        for ys in FTask_providers_match[eve.FTlist[0].ftask_id]:
                            jis=0
                            for ii in Sp[ys-1].workshop.equipments_state:
                                if ii != 0:
                                    jis += 1
                            minc.append(jis)
                        # indes=minc.index(min(minc))
                        indes=pos_in_lists(minc, min(minc))

                        indes_final=indes[0]
                        minds=count_path_on_road(eve.position, providers_positions[FTask_providers_match[eve.FTlist[0].ftask_id][indes_final]], 1)

                        if len(indes)!=1:
                            for evewe in indes:
                                ys=FTask_providers_match[eve.FTlist[0].ftask_id][evewe]
                                disys=count_path_on_road(eve.position, providers_positions[ys], 1)
                                if minds<disys:
                                    minds=disys
                                    indes_final=evewe
                        dict1[number]=FTask_providers_match[eve.FTlist[0].ftask_id][indes_final]

        return dict1

    if action=='MaxLoad_Mindistance_InOrder':#满足要求的里面，首个最小负载(二级),并且距离最远
        for eve in orders:
            for number in ordernumber_to_schedul:
                if eve.order_number == number:
                    if eve.position ==None:
                        choosed = random.choice(FTask_providers_match[eve.FTlist[0].ftask_id])
                        dict1[number] = choosed  # 0-,1-18
                    else:
                        minc=[]
                        for ys in FTask_providers_match[eve.FTlist[0].ftask_id]:
                            jis=0
                            for ii in Sp[ys-1].workshop.equipments_state:
                                if ii != 0:
                                    jis += 1
                            minc.append(jis)
                        # indes=minc.index(min(minc))
                        indes=pos_in_lists(minc, max(minc))

                        indes_final=indes[0]
                        minds=count_path_on_road(eve.position, providers_positions[FTask_providers_match[eve.FTlist[0].ftask_id][indes_final]], 1)

                        if len(indes)!=1:
                            for evewe in indes:
                                ys=FTask_providers_match[eve.FTlist[0].ftask_id][evewe]
                                disys=count_path_on_road(eve.position, providers_positions[ys], 1)
                                if minds<disys:
                                    minds=disys
                                    indes_final=evewe
                        dict1[number]=FTask_providers_match[eve.FTlist[0].ftask_id][indes_final]

        return dict1
    if action=='MaxLoad_Maxdistance_InOrder':#满足要求的里面，首个最小负载(二级),并且距离最远
        for eve in orders:
            for number in ordernumber_to_schedul:
                if eve.order_number == number:
                    if eve.position ==None:
                        choosed = random.choice(FTask_providers_match[eve.FTlist[0].ftask_id])
                        dict1[number] = choosed  # 0-,1-18
                    else:
                        minc=[]
                        for ys in FTask_providers_match[eve.FTlist[0].ftask_id]:
                            jis=0
                            for ii in Sp[ys-1].workshop.equipments_state:
                                if ii != 0:
                                    jis += 1
                            minc.append(jis)
                        # indes=minc.index(min(minc))
                        indes=pos_in_lists(minc, max(minc))

                        indes_final=indes[0]
                        minds=count_path_on_road(eve.position, providers_positions[FTask_providers_match[eve.FTlist[0].ftask_id][indes_final]], 1)

                        if len(indes)!=1:
                            for evewe in indes:
                                ys=FTask_providers_match[eve.FTlist[0].ftask_id][evewe]
                                disys=count_path_on_road(eve.position, providers_positions[ys], 1)
                                if minds>disys:
                                    minds=disys
                                    indes_final=evewe
                        dict1[number]=FTask_providers_match[eve.FTlist[0].ftask_id][indes_final]

        return dict1
    if action=='MinLoadrate_Mindistance_InOrder':#首个最小负载率(二级),并且距离最远
        for eve in orders:
            for number in ordernumber_to_schedul:
                if eve.order_number == number:
                    if eve.position ==None:
                        choosed = random.choice(FTask_providers_match[eve.FTlist[0].ftask_id])
                        dict1[number] = choosed  # 0-,1-18
                    else:
                        minc=[]
                        for ys in FTask_providers_match[eve.FTlist[0].ftask_id]:
                            jis=0
                            for ii in Sp[ys-1].workshop.equipments_state:
                                if ii != 0:
                                    jis += 1
                            minc.append(jis/len(Sp[ys-1].workshop.equipments_state))
                        # indes=minc.index(min(minc))
                        indes=pos_in_lists(minc, min(minc))

                        indes_final=indes[0]
                        minds=count_path_on_road(eve.position, providers_positions[FTask_providers_match[eve.FTlist[0].ftask_id][indes_final]], 1)

                        if len(indes)!=1:
                            for evewe in indes:
                                ys=FTask_providers_match[eve.FTlist[0].ftask_id][evewe]
                                disys=count_path_on_road(eve.position, providers_positions[ys], 1)
                                if minds<disys:
                                    minds=disys
                                    indes_final=evewe
                        dict1[number]=FTask_providers_match[eve.FTlist[0].ftask_id][indes_final]

        return dict1

    if action=='Minwaiting_InOrder':#首个等待率(二级),并且距离最远
        for eve in orders:
            for number in ordernumber_to_schedul:
                if eve.order_number == number:
                    if eve.position ==None:
                        choosed = random.choice(FTask_providers_match[eve.FTlist[0].ftask_id])
                        dict1[number] = choosed  # 0-,1-18
                    else:
                        minc=[]
                        for ys in FTask_providers_match[eve.FTlist[0].ftask_id]:
                            minc.append(len(Sp[ys-1].wait_beserved_ST))
                        # indes=minc.index(min(minc))
                        indes=pos_in_lists(minc, min(minc))

                        indes_final=indes[0]
                        minds=count_path_on_road(eve.position, providers_positions[FTask_providers_match[eve.FTlist[0].ftask_id][indes_final]], 1)

                        if len(indes)!=1:
                            for evewe in indes:
                                ys=FTask_providers_match[eve.FTlist[0].ftask_id][evewe]
                                disys=count_path_on_road(eve.position, providers_positions[ys], 1)
                                if minds>disys:
                                    minds=disys
                                    indes_final=evewe
                        dict1[number]=FTask_providers_match[eve.FTlist[0].ftask_id][indes_final]

        return dict1


    if action=='MinDistance_InOrder':
        for eve in orders:
            for number in ordernumber_to_schedul:
                if eve.order_number == number:
                    if eve.position ==None:
                        choosed = random.choice(FTask_providers_match[eve.FTlist[0].ftask_id])
                        dict1[number] = choosed  # 0-,1-18
                    else:
                        minc=[]
                        for ys in FTask_providers_match[eve.FTlist[0].ftask_id]:
                            minc.append(count_path_on_road(eve.position, providers_positions[ys], 1))
                        indes=minc.index(min(minc))
                        dict1[number]=FTask_providers_match[eve.FTlist[0].ftask_id][indes]
                          # 0-,1-18
        return dict1

    if action=='MaxDistance_InOrder':
        for eve in orders:
            for number in ordernumber_to_schedul:
                if eve.order_number == number:
                    if eve.position ==None:
                        choosed = random.choice(FTask_providers_match[eve.FTlist[0].ftask_id])
                        dict1[number] = choosed  # 0-,1-18
                    else:
                        minc=[]
                        for ys in FTask_providers_match[eve.FTlist[0].ftask_id]:
                            minc.append(count_path_on_road(eve.position, providers_positions[ys], 1))
                        indes=minc.index(max(minc))
                        dict1[number]=FTask_providers_match[eve.FTlist[0].ftask_id][indes]
                          # 0-,1-18
        return dict1



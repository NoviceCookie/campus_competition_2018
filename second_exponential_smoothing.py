import os
import pandas as pd
import no_holiday
import datetime

def predict_three(data,beta=0.2):
    '''
    三次平滑预测
    :param data:
    :param beta:
    :return:
    '''
    s1 = []
    s2 = []
    s3 = []
    MSE = 0
    s = float(sum(data[:3])) / 3
    s1.append(s)
    s2.append(s)
    s3.append(s)
    for i in range(len(data)):
        s = beta * data[i] + (1 - beta) * s1[i]
        s1.append(s)
    # print(s1)
    for i in range(1, len(s1)):
        s = beta * s1[i] + (1 - beta) * s2[i - 1]
        s2.append(s)

    for i in range(1, len(s2)):
        s = beta * s2[i] + (1 - beta) * s3[i - 1]
        s3.append(s)
        At = float(s1[i - 1]) * 3 -3*float(s2[i-1])+float(s3[i - 1])
        Bt = float(beta) / (2*(1 - float(beta))**2) *\
             ((6-5*beta)*s1[i - 1]-2*(5-4*beta)*s2[i - 1]+(4-3*beta)*s3[i-1])
        Ct = beta**2/(2*(1-beta)**2)*(s1[i-1]-2*s2[i-1]+s3[i-1])
        pre = int(At + Bt + Ct + 0.5)
        # pre = s2[i]
        MSE = (pre - int(data[i - 1])) ** 2 + MSE


    # print(s2)

    At = float(s1[-1]) * 3 - 3 * float(s2[-1]) + float(s3[-1])
    Bt = float(beta) / (2 * (1 - float(beta)) ** 2) * \
         ((6 - 5 * beta) * s1[-1] - 2 * (5 - 4 * beta) * s2[-1] + (4 - 3 * beta) * s3[-1])
    Ct = beta ** 2 / (2 * (1 - beta) ** 2) * (s1[-1] - 2 * s2[-1] + s3[-1])
    result = int(At + Bt + Ct + 0.5)
    MSE = (MSE ** (1 / 2)) / len(data)
    # print(At,' at --- bt ',Bt)

    return result, MSE, s1, s2,s3


def predict(data,beta=0.2):
    '''
    二次平滑预测
    :param data:
    :param beta:
    :return:
    '''
    s1 = []
    s2 = []
    MSE = 0
    s = float(sum(data[:3]))/3
    s1.append(s)
    s2.append(s)
    for i in range(len(data)):
        s = beta*data[i]+(1-beta)*s1[i]
        s1.append(s)
    # print(s1)
    for i in range(1,len(s1)):
        s = beta * s1[i] + (1 - beta) * s2[i-1]
        s2.append(s)
        At = float(s1[i-1]) * 2 - float(s2[i-1])
        Bt = float(beta) / (1 - float(beta)) * (s1[i-1] - s2[i-1])
        pre = int(At+Bt+0.5)
        # pre = s2[i]
        MSE = ( pre- int(data[i-1]))**2 + MSE
    # print(s2)
    At = float(s1[-1])*2 - float(s2[-1])
    Bt = float(beta)/(1-float(beta))*(s1[-1]-s2[-1])
    result = int(At+Bt+0.5)
    MSE = (MSE**(1/2))/len(data)
    # print(At,' at --- bt ',Bt)

    return result,MSE,s1,s2



def predict_loc_three(frame,time_stamp,loc_id):
    '''
    周对应星期时间预测,三次平滑，遍历调参,输出平滑序列
    :param frame:
    :param time_stamp:
    :param loc_id:
    :return:
    '''
    data = no_holiday.get_week_data(frame,time_stamp,loc_id)
    if len(data)==0:
        return 0,0,0
    elif len(data)<3:
        return int(sum(data)//len(data)),0,0

    mse = 1000000
    best = sum(data)//len(data)
    bestBeta =0.01
    s1,s2,s3=[],[],[]
    for i in range(20,80):
        beta = i * 0.01
        temp,err,s_temp1,s_temp2,s_temp3 = predict_three(data,beta)
        # print(temp,err)
        if err<mse:
            best = temp
            bestBeta = beta
            mse = err
            s1 = s_temp1
            s2 = s_temp2
            s3 = s_temp3

    # print(best,err,bestBeta)
    aver = sum(data) // len(data)
    if best<0:
        best = aver
        print('less than zero')
    if best>aver*3:
        best = int(aver*3)
        print('bigger than 3 times of average')

    #输出data，平滑序列，s1,s2
    data_s = ','.join([str(x) for x in data])+','+str(best) + '\n'
    s1_s = ','.join([str(x) for x in s1[1:]]) + '\n'
    s2_s = ','.join([str(x) for x in s2[1:]]) + '\n'
    s3_s = ','.join([str(x) for x in s3[1:]]) + '\n'
    with open('dat_s1_s2_s3_olderror_v2.csv', 'a', encoding='utf-8') as f:
        f.write(data_s)
        f.write(s1_s)
        f.write(s2_s)
        f.write(s3_s)


    return best,err,bestBeta

def predict_loc_december(frame,time_stamp,loc_id,data_s1_s2='data_s1_s2_december_last.csv'):
    return predict_loc3(frame,time_stamp,loc_id,data_s1_s2)

def predict_loc_december_thr_months(frame,time_stamp,loc_id,data_s1_s2='data_s1_s2_thr_months_v3.csv'):
    '''
    周对应星期时间，二次平滑，遍历调参,输出平滑序列
    :param frame:
    :param time_stamp:
    :param loc_id:
    :param data_s1_s2:
    :return:
    '''
    data = no_holiday.get_week_3months_data(frame,time_stamp,loc_id)
    if len(data)==0:
        return 0,0,0
    elif len(data)<3:
        return int(sum(data)//len(data)),0,0

    mse = 1000000
    best = sum(data)//len(data)
    bestBeta =0.01
    s1,s2=[],[]
    for i in range(10,80):
        beta = i * 0.01
        temp,err,s_temp1,s_temp2  = predict(data,beta)
        # print(temp,err)
        if err<mse:
            best = temp
            bestBeta = beta
            mse = err
            s1 = s_temp1
            s2 = s_temp2

    # print(best,err,bestBeta)
    aver = sum(data) // len(data)
    if best<0:
        print('best : ',best)
        best = 0
        print('less than zero: ',best)
    if best>aver*3:
        print('best : ', best)
        best = int(aver*3)
        print('bigger than 3 times of average: ',best)

    #输出data，平滑序列，s1,s2
    data_s = ','.join([str(x) for x in data])+','+str(best) + '\n'
    s1_s = ','.join([str(x) for x in s1[1:]]) + '\n'
    s2_s = ','.join([str(x) for x in s2[1:]]) + '\n'
    with open(data_s1_s2, 'a', encoding='utf-8') as f:
        f.write(data_s)
        f.write(s1_s)
        f.write(s2_s)
    return best,err,bestBeta

def predict_loc3(frame,time_stamp,loc_id,data_s1_s2='data_s1_s2_v1.csv'):
    '''
    周对应星期时间，二次平滑，遍历调参,输出平滑序列
    :param frame:
    :param time_stamp:
    :param loc_id:
    :param data_s1_s2:
    :return:
    '''
    data = no_holiday.get_week_data(frame,time_stamp,loc_id)
    if len(data)==0:
        return 0,0,0
    elif len(data)<3:
        return int(sum(data)//len(data)),0,0

    mse = 1000000
    best = sum(data)//len(data)
    bestBeta =0.01
    s1,s2=[],[]
    for i in range(10,80):
        beta = i * 0.01
        temp,err,s_temp1,s_temp2  = predict(data,beta)
        # print(temp,err)
        if err<mse:
            best = temp
            bestBeta = beta
            mse = err
            s1 = s_temp1
            s2 = s_temp2

    # print(best,err,bestBeta)
    aver = sum(data) // len(data)
    if best<aver/4:
        best = int(aver/4)
        print('less than zero')
    if best>aver*3:
        best = int(aver*3)
        print('bigger than 3 times of average')

    #输出data，平滑序列，s1,s2
    data_s = ','.join([str(x) for x in data])+','+str(best) + '\n'
    s1_s = ','.join([str(x) for x in s1[1:]]) + '\n'
    s2_s = ','.join([str(x) for x in s2[1:]]) + '\n'
    with open(data_s1_s2, 'a', encoding='utf-8') as f:
        f.write(data_s)
        f.write(s1_s)
        f.write(s2_s)
    return best,err,bestBeta


def predict_loc2(frame,time_stamp,loc_id,data_s1_s2='data_s1_s2_v1.csv'):
    '''
     月对应周次星期时间预测，二次平滑，遍历调参,输出平滑序列
    :param frame:
    :param time_stamp:
    :param loc_id:
    :param data_s1_s2:
    :return:
    '''
    data = no_holiday.get_data(frame,time_stamp,loc_id)
    print('==',data)
    if len(data)==0:
        return 0,0,0
    elif len(data)<3:
        return int(sum(data)//len(data)),0,0

    mse = 1000000
    best = sum(data)//len(data)
    bestBeta =0.1
    s1, s2 = [], []
    for i in range(1,80):
        beta = i*0.01
        temp,err,s_temp1,s_temp2  = predict(data,beta)
        # print(temp,err)
        if err<mse:
            best = temp
            bestBeta = beta
            mse = err
            s1 = s_temp1
            s2 = s_temp2
    # print(best,err,bestBeta)
    aver = sum(data) // len(data)
    if best<(aver/4):
        best = int(aver/4)
        print('less than zero')
    if best>aver*3:
        best = int(aver*3)
        print('bigger than 3 times of average')

    # 输出data，平滑序列，s1,s2
    data_s = ','.join([str(x) for x in data]) + ',' + str(best) + '\n'
    s1_s = ','.join([str(x) for x in s1[1:]]) + '\n'
    s2_s = ','.join([str(x) for x in s2[1:]]) + '\n'
    with open(data_s1_s2, 'a', encoding='utf-8') as f:
        f.write(data_s)
        f.write(s1_s)
        f.write(s2_s)

    return best,err,bestBeta


def predict_loc(local_id,time,df):
    '''
    二次平滑，遍历调参
    :param local_id:
    :param time:
    :param df:
    :return:
    '''
    dataf = df[df['time_stamp'].str.contains(time)]
    # print(dataf)
    dataf = dataf[dataf['loc_id']==local_id]
    dataf = dataf.sort_values(by=['time_stamp']).reset_index(drop=True)
    print(local_id,' == dataf')
    print(dataf)
    data = dataf['num_of_people'].tolist()
    print(data)
    if len(data)==0:
        return 0,0,0
    elif len(data)<3:
        return int(sum(data)//len(data)),0,0

    mse = 1000000
    best = sum(data)//len(data)
    bestBeta =0.1
    beta = 0.1

    for i in range(9):
        temp,err  = predict(data,beta)
        # print(temp,err)
        if err<mse:
            best = temp
            bestBeta = beta
            mse = err
        beta =beta + 0.1
    # print(best,err,bestBeta)
    aver = sum(data) // len(data)
    if best<0:
        best = aver
        print('less than zero')
    if best>aver*3:
        best = int(aver*3)
        print('bigger than 3 times of average')
    return best,err,bestBeta

def predict_All_three():
    '''
    三次指数平滑，预测11月上网人数
    :return:
    '''
    result = pd.DataFrame(columns=['loc_id', 'time_stamp', 'num_of_people'])
    # dir = 'E:/school_compet/no_holiday_all_3.csv'
    dir = 'E:/school_compet/no_holiday_all_3.csv'
    frame = pd.read_csv(dir, header=0, sep=',')
    result_temp =[[]for i in range(7)]
    pare = open('err_beta_three_old_err_v2.txt','w',encoding='utf-8')
    for d in range(1, 8):
        for loc in range(1, 34):
            for t in range(24):
                # 预测特定地点，时间戳的人数
                dtime = datetime.datetime(2017, 11, d, t)
                best, err, beta = predict(frame, dtime, loc)
                out_record = 'local:%d ,day:%d, time:%d, err:%f, beta:%f\n'\
                            %(loc,d,t,err,beta)
                pare.write(out_record)
                result_temp[d-1].append([loc,t,best])
    pare.close()

    for d in range(1,31):
        d_date = result_temp[(d-1)%7]
        for x in d_date:
            dtime = datetime.date(2017, 11, d)
            time_stamp = str(dtime) + ' %02d' % x[1]
            # print(time_stamp)
            news = pd.DataFrame({'loc_id': x[0], 'time_stamp': time_stamp, 'num_of_people': x[2]}, index=['0'])
            result = result.append(news, ignore_index=True)

    result = result.sort_values(by=['time_stamp','loc_id']).reset_index(drop=True)
    col = ['loc_id', 'time_stamp', 'num_of_people']
    result.to_csv('E:/school_compet/final_result_week_old_err_three_v2.csv', columns=col, index=False, sep=',')

def predict_all_december_months_week(err_beta='err_beta_december_month_week.txt', result_december='result_december_month_week.csv'):
    '''
    利用前11月数据，根据月对应周次的星期数，预测12月上网人数
    :param err_beta:
    :param result_december:
    :return:
    '''
    result = pd.DataFrame(columns=['loc_id', 'time_stamp', 'num_of_people'])
    # dir = 'E:/school_compet/no_holiday_all_3.csv'
    dir = 'E:/school_compet/final_data/holiday_fix_diff_all.csv'
    frame = pd.read_csv(dir,header=0,sep=',')
    pare = open(err_beta, 'w', encoding='utf-8')
    for loc in range(1,34):
        for d in range(1,32):
            for t in range(24):

                # 预测特定地点，时间戳的人数
                dtime = datetime.datetime(2017, 12, d, t)
                best, err, beta = predict_loc2(frame,dtime ,loc,'data_s1_s2_month_week.csv')
                out_record = 'local:%d ,day:%d, time:%d, err:%f, beta:%f\n' \
                             % (loc, d, t, err, beta)
                pare.write(out_record)
                time_stamp = str(dtime.date())+' %02d'%t
                print(time_stamp)
                news = pd.DataFrame({'loc_id': loc, 'time_stamp': time_stamp, 'num_of_people': best}, index=['0'])
                result = result.append(news, ignore_index=True)

    result = result.sort_values(by=['time_stamp','loc_id']).reset_index(drop=True)
    col = ['loc_id', 'time_stamp', 'num_of_people']
    result.to_csv(result_december, columns=col, index=False, sep=',')


def predict_all_decmber_last_thr_months(err_beta='err_beta_december_thr_months.txt', result_december='result_december_thr_months_last_v1.csv'):
    '''
    利用9,10,11月数据，周对应星期数，预测12月上网数据
    :param err_beta:
    :param result_december:
    :return:
    '''
    result = pd.DataFrame(columns=['loc_id', 'time_stamp', 'num_of_people'])
    # dir = 'E:/school_compet/no_holiday_all_3.csv'
    ########################3
    #########################
    dir = 'E:/school_compet/final_data/holiday_fix_diff_all_last_v.csv'
    frame = pd.read_csv(dir, header=0, sep=',')
    result_temp = [[] for i in range(7)]
    pare = open(err_beta, 'w', encoding='utf-8')
    for d in range(1, 8):
        for loc in range(1, 34):
            for t in range(24):
                # 预测特定地点，时间戳的人数
                dtime = datetime.datetime(2017, 12, d, t)
                best, err, beta = predict_loc_december_thr_months(frame, dtime, loc)
                out_record = 'local:%d ,day:%d, time:%d, err:%f, beta:%f\n' \
                             % (loc, d, t, err, beta)
                pare.write(out_record)
                result_temp[d - 1].append([loc, t, best])
    pare.close()

    for d in range(1, 30):
        d_date = result_temp[(d - 1) % 7]
        for x in d_date:
            dtime = datetime.date(2017, 12, d)
            time_stamp = str(dtime) + ' %02d' % x[1]
            # print(time_stamp)
            news = pd.DataFrame({'loc_id': x[0], 'time_stamp': time_stamp, 'num_of_people': x[2]}, index=['0'])
            result = result.append(news, ignore_index=True)
    #假期权值计算
    weightfile = 'E:/school_compet/final_data/holiday_weight_last_v.csv'
    weight_data = pd.read_csv(weightfile, header=0, sep=',')
    weight = [[] for x in range(33)]
    for loc in range(1, 34):
        for t in range(24):
            wtemp = weight_data[(weight_data['loc_id'] == loc) & (weight_data['hour'] == t)]
            # print(wtemp)
            wcount = wtemp[wtemp['month'] == 1]['weight'].iloc[0] * 0.4 + wtemp[wtemp['month'].isin([4, 5, 10])][
                                                                              'weight'].sum() * 0.2
            print('wcount: ', wcount)
            weight[loc - 1].append(wcount)
    for d in range(30,32):
        d_date = result_temp[(d-1)%7]
        for x in d_date:
            dtime = datetime.date(2017, 12, d)
            time_stamp = str(dtime) + ' %02d' % x[1]
            # print(time_stamp)
            news = pd.DataFrame({'loc_id': x[0], 'time_stamp': time_stamp, 'num_of_people': int(x[2]*weight[x[0]-1][x[1]])}, index=['0'])
            result = result.append(news, ignore_index=True)

    result = result.sort_values(by=['time_stamp', 'loc_id']).reset_index(drop=True)
    col = ['loc_id', 'time_stamp', 'num_of_people']
    result.to_csv(result_december, columns=col, index=False, sep=',')


def predict_all_december(err_beta='err_beta_december.txt',result_december='result_december_last_v1.csv'):
    '''
    修正假期数据，周对应星期数，预测12月上网人数（包含最后两天假期，比例修正）
    :param err_beta:
    :param result_december:
    :return:
    '''
    result = pd.DataFrame(columns=['loc_id', 'time_stamp', 'num_of_people'])
    # dir = 'E:/school_compet/no_holiday_all_3.csv'
    ########################3
    #########################
    dir = 'E:/school_compet/final_data/holiday_fix_diff_all_last_v.csv'
    frame = pd.read_csv(dir, header=0, sep=',')
    result_temp = [[] for i in range(7)]
    pare = open(err_beta, 'w', encoding='utf-8')
    for d in range(1, 8):
        for loc in range(1, 34):
            for t in range(24):
                # 预测特定地点，时间戳的人数
                dtime = datetime.datetime(2017, 12, d, t)
                best, err, beta = predict_loc_december(frame, dtime, loc)
                out_record = 'local:%d ,day:%d, time:%d, err:%f, beta:%f\n' \
                             % (loc, d, t, err, beta)
                pare.write(out_record)
                result_temp[d - 1].append([loc, t, best])
    pare.close()

    for d in range(1, 30):
        d_date = result_temp[(d - 1) % 7]
        for x in d_date:
            dtime = datetime.date(2017, 12, d)
            time_stamp = str(dtime) + ' %02d' % x[1]
            # print(time_stamp)
            news = pd.DataFrame({'loc_id': x[0], 'time_stamp': time_stamp, 'num_of_people': x[2]}, index=['0'])
            result = result.append(news, ignore_index=True)
    #假期权值计算
    weightfile = 'E:/school_compet/final_data/holiday_weight_last_v.csv'
    weight_data = pd.read_csv(weightfile, header=0, sep=',')
    weight = [[] for x in range(33)]
    for loc in range(1, 34):
        for t in range(24):
            wtemp = weight_data[(weight_data['loc_id'] == loc) & (weight_data['hour'] == t)]
            # print(wtemp)
            wcount = wtemp[wtemp['month'] == 1]['weight'].iloc[0] * 0.4 + wtemp[wtemp['month'].isin([4, 5, 10])][
                                                                              'weight'].sum() * 0.2
            print('wcount: ', wcount)
            weight[loc - 1].append(wcount)
    for d in range(30,32):
        d_date = result_temp[(d-1)%7]
        for x in d_date:
            dtime = datetime.date(2017, 12, d)
            time_stamp = str(dtime) + ' %02d' % x[1]
            # print(time_stamp)
            news = pd.DataFrame({'loc_id': x[0], 'time_stamp': time_stamp, 'num_of_people': int(x[2]*weight[x[0]-1][x[1]])}, index=['0'])
            result = result.append(news, ignore_index=True)

    result = result.sort_values(by=['time_stamp', 'loc_id']).reset_index(drop=True)
    col = ['loc_id', 'time_stamp', 'num_of_people']
    result.to_csv(result_december, columns=col, index=False, sep=',')


def predict_All3():
    #去除假期，周对应星期数，预测11月上网日期
    result = pd.DataFrame(columns=['loc_id', 'time_stamp', 'num_of_people'])
    # dir = 'E:/school_compet/no_holiday_all_3.csv'
    dir = 'E:/school_compet/no_holiday_all_3.csv'
    frame = pd.read_csv(dir, header=0, sep=',')
    result_temp =[[]for i in range(7)]
    pare = open('err_beta_old_err.txt','w',encoding='utf-8')
    for d in range(1, 8):
        for loc in range(1, 34):
            for t in range(24):
                # 预测特定地点，时间戳的人数
                dtime = datetime.datetime(2017, 11, d, t)
                best, err, beta = predict_loc3(frame, dtime, loc)
                out_record = 'local:%d ,day:%d, time:%d, err:%f, beta:%f\n'\
                            %(loc,d,t,err,beta)
                pare.write(out_record)
                result_temp[d-1].append([loc,t,best])
    pare.close()

    for d in range(1,31):
        d_date = result_temp[(d-1)%7]
        for x in d_date:
            dtime = datetime.date(2017, 11, d)
            time_stamp = str(dtime) + ' %02d' % x[1]
            # print(time_stamp)
            news = pd.DataFrame({'loc_id': x[0], 'time_stamp': time_stamp, 'num_of_people': x[2]}, index=['0'])
            result = result.append(news, ignore_index=True)

    result = result.sort_values(by=['time_stamp','loc_id']).reset_index(drop=True)
    col = ['loc_id', 'time_stamp', 'num_of_people']
    result.to_csv('E:/school_compet/final_result_week_old_err_test_v2.csv', columns=col, index=False, sep=',')

def predict_All2():
    #去除假期，月对应日期，预测11月上网人数
    result = pd.DataFrame(columns=['loc_id', 'time_stamp', 'num_of_people'])
    # dir = 'E:/school_compet/no_holiday_all_3.csv'
    dir = 'E:/school_compet/no_holiday_loc_diff.csv'
    frame = pd.read_csv(dir,header=0,sep=',')
    for loc in range(1,34):
        for d in range(1,31):
            for t in range(24):

                # 预测特定地点，时间戳的人数
                dtime = datetime.datetime(2017, 11, d, t)
                best, err, beta = predict_loc2(frame,dtime ,loc)

                time_stamp = str(dtime.date())+' %02d'%t
                print(time_stamp)
                news = pd.DataFrame({'loc_id': loc, 'time_stamp': time_stamp, 'num_of_people': best}, index=['0'])
                result = result.append(news, ignore_index=True)

    result = result.sort_values(by=['time_stamp','loc_id']).reset_index(drop=True)
    col = ['loc_id', 'time_stamp', 'num_of_people']
    result.to_csv('E:/school_compet/final_result_moth_week_v1.csv', columns=col, index=False, sep=',')


def predict_All():
    #月对应日期，预测11月上网人数
    print('predict_all')
    result = pd.DataFrame(columns=['loc_id','time_stamp','num_of_people'])
    dir = 'E:/school_compet/no_holiday'

    time = ['%02d' % x for x in range(24)]
    date = ['%02d' % x for x in range(1,31)]
    fileList = os.listdir(dir)
    fileList.sort(key=lambda x:int(x.split('_')[0]))
    print(fileList)
    for file in fileList:
        fileName = os.path.join(dir,file)
        print(file)
        local = file.split('_')[0]
        print('local site: ',local)
        data = pd.read_csv(fileName,header=0,sep=',')
        for d in date:
            for t in time:
                time_stamp = '-'+d+' '+t
                print(time_stamp)
                #预测特定地点，时间戳的人数
                best,err,beta = predict_loc(int(local),time_stamp,data)
                # print('local:',local,' date:2017-10'+time_stamp,' num:',best,' err:',err,' beta:',beta)
                time_stamp = '2017-11'+time_stamp
                news = pd.DataFrame({'loc_id':local,'time'
                                                    '_stamp':time_stamp,'num_of_people':best},index=['0'])
                result = result.append(news,ignore_index=True)

    # result = result.sort_values(by=['loc_id','time_stamp']).reset_index(drop=True)
    col = ['loc_id', 'time_stamp', 'num_of_people']
    result.to_csv('E:/school_compet/final_result_v9.0.csv',columns=col,index=False,sep=',')

if __name__ == '__main__':
    print('ok')
    # predict_All_three()
    # predict_all_december()
    predict_all_decmber_last_thr_months()
    # predict_all_december_months_week()

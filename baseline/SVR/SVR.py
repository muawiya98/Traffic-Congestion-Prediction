# -- coding: utf-8 --
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.svm import SVR
import os
import tqdm


# file='/home/ibdi_public/traffic/copy/3S-TBLN/data/metr-la/'
data_path = "E:\\Tasks\\09. Traffic Congestion Prediction\\Traffic Congestion Prediction\\Dataset"
root_path = 'E:\\Tasks\\09. Traffic Congestion Prediction\\STGIN-main'
main_results_path = os.path.join(root_path, 'Results_R')


class svm_i():
    def __init__(self,
                 site_id=0,
                 is_training=True,
                 data_divide=0.7,
                 window_step=1,
                 normalize=False,
                 folder_name='',
                 data_name='',
                 r_ratios=1,
                 Horizon=12,
                 site_num=0
                 ):
        '''
        :param is_training: while is_training is True,the model is training state
        :param field_len:
        :param time_size:
        :param prediction_size:
        :param target_site:
        '''
        if not os.path.exists(main_results_path): os.makedirs(main_results_path)
        self.save_path = os.path.join(main_results_path, 'weights_'+folder_name+'_SVM_R_ratio'+str(r_ratios))
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)
        self.save_path = os.path.join(self.save_path, 'weights_'+folder_name+'_'+'SVM_R_ratio'+str(r_ratios))
        self.results_path = os.path.join(main_results_path, 'Results_'+folder_name+'_SVM_R_ratio'+str(r_ratios))
        if not os.path.exists(self.results_path): os.makedirs(self.results_path)
        
        self.file = os.path.join(data_path, folder_name)
        self.Horizon = Horizon
        self.r_ratios = r_ratios
        time_size=Horizon,
        prediction_size=Horizon,
        self.site_num = site_num
        
        self.site_id=site_id                   # ozone ID
        self.time_size=time_size               # time series length of input
        self.prediction_size=prediction_size   # the length of prediction
        self.is_training=is_training           # true or false
        self.data_divide=data_divide           # the divide between in training set and test set ratio
        self.window_step=window_step           # windows step
        self.data=self.get_source_data(os.path.join(self.file, data_name))
        self.data = self.data.loc[:3500*site_num]
        self.length=self.data.values.shape[0]  #data length

    def get_source_data(self,file_path):
        '''
        :return:
        '''
        data = pd.read_csv(file_path, encoding='utf-8')
        return data

    def describe(self, label, predict):
        '''
        :param label:
        :param predict:
        :param prediction_size:
        :return:
        '''
        plt.figure()
        # Label is observed value,Blue
        plt.plot(label[0:], 'b', label=u'actual value')
        # Predict is predicted value，Red
        plt.plot(predict[0:], 'r', label=u'predicted value')
        # use the legend
        plt.legend()
        # plt.xlabel("time(hours)", fontsize=17)
        # plt.ylabel("pm$_{2.5}$ (ug/m$^3$)", fontsize=17)
        # plt.title("the prediction of pm$_{2.5}", fontsize=17)
        plt.show()

    def metric(self, pred, label):
        with np.errstate(divide='print', invalid='ignore'):
            mask = np.not_equal(label, 0)
            mask = mask.astype(np.float32)
            mask /= np.mean(mask)

            mae = np.abs(np.subtract(pred, label)).astype(np.float32)
            rmse = np.square(mae)
            mape = np.divide(mae, label.astype(np.float32))
            mae = np.nan_to_num(mae * mask)
            mae = np.mean(mae)
            rmse = np.nan_to_num(rmse * mask)
            rmse = np.sqrt(np.mean(rmse))
            mape = np.nan_to_num(mape * mask)
            mape = np.mean(mape)
            cor = np.mean(np.multiply((label - np.mean(label)),
                                    (pred - np.mean(pred)))) / (np.std(pred) * np.std(label))
            sse = np.sum((label - pred) ** 2)
            sst = np.sum((label - np.mean(label)) ** 2)
            r2 = 1 - sse / sst  # r2_score(y_actual, y_predicted, multioutput='raw_values')
        return mae, rmse, mape

    def train_data(self,data,input_length,predict_length):
        low,high=0,data.shape[0]
        x ,y=[],[]
        while low+predict_length+input_length<high:
            x.append(np.reshape(data[low:low+input_length],newshape=[-1]))
            y.append(np.reshape(data[low+input_length:low+input_length+predict_length],newshape=[-1]))
            low+=self.window_step
        return np.array(x), np.array(y)

    def model(self):
        toll_predict, toll_label = list(), list()
        t_maes, t_rmses, t_mapes, maes, rmses, mapes, steps_list = [], [], [], [], [], [], []
        print('                MAE\t\tRMSE\t\tMAPE')
        for time_step in range(self.Horizon):
            print('current step is ',time_step)
            labels, train_labels, predicts, train_predicts = [], [], [], []

            predict_index=time_step
            for site in range(0, self.site_num):
            # for site in segment:
                data1=self.data[(self.data['node']==self.data.values[site][0])]
                x = data1.values[:, -1]

                x, y=self.train_data(data=x,input_length=self.Horizon, predict_length=self.Horizon)

                train_size = int(len(x) * 0.7)
                val_size = int
                test_size = int(len(x) * 0.2)
                
                train_x, train_y, test_x, test_y = x[:train_size], y[:train_size, predict_index], x[-test_size:], y[-test_size:,predict_index]

                r_ratio_ = np.random.randint(0, train_x.shape[0], self.r_ratios)
                train_x[r_ratio_] = 0
                # print(train_x.shape, train_y.shape, test_x.shape,test_y.shape)
                # print(data1.shape)
                # svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                #                    param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                #                                "gamma": np.logspace(-2, 2, 5)})

                svr = SVR(C=4, degree=2)

                # model=svm.NuSVR(nu=0.457,C=.8,degree=3)
                svr.fit(X=train_x, y=train_y)

                
                pre = svr.predict(X=test_x)
                train_pre = svr.predict(X=train_x)
                
                predicts.append(np.expand_dims(pre, axis=1))
                train_predicts.append(np.expand_dims(train_pre, axis=1))
                
                labels.append(np.expand_dims(test_y,axis=1))
                train_labels.append(np.expand_dims(train_y,axis=1))

            mae, rmse, mape = self.metric(np.concatenate(predicts), np.concatenate(labels))
            t_mae, t_rmse, t_mape = self.metric(np.concatenate(predicts), np.concatenate(labels))

            print('step: %02d         %.3f\t\t%.3f\t\t%.3f%%' % (time_step + 1, mae, rmse, mape * 100))
            steps_list.append(f"step {time_step + 1}")
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)

            t_maes.append(t_mae)
            t_rmses.append(t_rmse)
            t_mapes.append(t_mape)

        # self.metric(np.reshape(np.array(toll_predict), newshape=[-1]), np.reshape(np.array(toll_label), newshape=[-1]))
        print('average:         %.3f\t\t%.3f\t\t%.3f%%' % (np.array(maes).mean(), np.array(rmses).mean(), np.array(mapes).mean() * 100))
        steps_list.append("average")
        mae_list = maes + [np.array(maes).mean()]
        rmse_list = rmses + [np.array(rmses).mean()]
        mape_list = mapes + [np.array(mapes).mean()]
        df = pd.DataFrame(columns=['steps', 'MAE', 'RMSE', 'MAPE'])
        df['steps'], df['MAE'], df['RMSE'], df['MAPE'] = steps_list, mae_list, rmse_list, mape_list
        df.to_csv(os.path.join(self.results_path, 'test results.csv'), index=False)

        mae_list = t_maes + [np.array(t_maes).mean()]
        rmse_list = t_rmses + [np.array(t_rmses).mean()]
        mape_list = t_mapes + [np.array(t_mapes).mean()]
        df = pd.DataFrame(columns=['MAE', 'RMSE', 'MAPE'])
        df['MAE'], df['RMSE'], df['MAPE'] = mae_list, rmse_list, mape_list
        df.to_csv(os.path.join(self.results_path, 'train results.csv'), index=False)

#
if __name__=='__main__':


    folder_name = ['METRLA', 'PEMSBAY']
    data_name = ['M_Metrla.csv', 'M_Pemsbay.csv'] # M_Metrla, M_Pemsbay
    site_num = [207, 325]
    Horizon = [3, 6, 12]
    r_ratios = [10, 5, 1]
    

    # train run
    ha=svm_i(site_id=0, normalize=False, folder_name='METRLA', 
             data_name='M_Metrla.csv', r_ratios=10, Horizon=12, site_num=207)
    
    # test run
    # ha=svm_i(site_id=0, normalize=False, folder_name='METRLA', 
    #         data_name='M_Metrla.csv', r_ratios=10, Horizon=12, site_num=207)
    
    ha.model()
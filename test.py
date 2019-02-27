import numpy as np
import matplotlib.pyplot as plt 
import h5py
import scipy
from PIL import Image 
from scipy import ndimage
from lr_utils import load_dataset

#读入数据库，如图像，训练集，测试集
train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes=load_dataset()


#此乃训练集数量
m_train=train_set_x_orig.shape[0]
#此乃测试集数量
m_test=train_set_x_orig.shape[0]
#此乃图像的点阵规模
num_px=train_set_x_orig.shape[1]
#将矩阵标准化
train_set_x_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

train_set_x=train_set_x_flatten/255
test_set_x=test_set_x_flatten/255

#sigmoid函数，亦称为逻辑回归函数
def sigmoid(z):
	s=1/(1+np.exp(-z))
	return s
#print("sigmoid"+str(sigmoid(np.array([0,2]))))


def init(dim):
	w=np.zeros((dim,1))
	b=0
	return w,b

#前向传播
def pro(w,b,X,Y):
	m=X.shape[1]
	A=sigmoid(np.dot(w.T ,X)+b)
	#损失函数
	cost=(-1/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
	#此处记录梯度
	dw=(1/m)*np.dot(X,(A-Y).T)
	db=(1/m)*np.sum(A-Y)
	cost=np.squeeze(cost)
	#梯度cache
	grads={"dw":dw,"db":db}

	return grads,cost
#反向传播和梯度下降
def opt(w,b,X,Y,num_ite,learning_rate,print_cost=False):
	costs=[]
	#不断迭代进行梯度下降
	for i in range(num_ite):
		grads,cost=pro(w,b,X,Y)
		dw=grads["dw"]
		db=grads["db"]
		w=w-learning_rate*dw 
		b=b-learning_rate*db 
		#每100次迭代输出损失函数作为对比
		if i%100==0:
			costs.append(cost)
		if print_cost and i%100==0:
			print("Cost after iteration %i:%f"%(i,cost))

	params={"w":w,"b":b}
	grads={"dw":dw,"db":db}
	return params,grads,costs 
#预测函数
def predict(w,b,X):
	m=X.shape[1]
	Y_prediction=np.zeros((1,m))
	w=w.reshape(X.shape[0],1)

	A=sigmoid(np.dot(w.T,X)+b)
#二元分类
	for i in range(A.shape[1]):
		if A[0,i]>0.5 :
			Y_prediction[0,i]=1
		else:
			Y_prediction[0,i]=0

	return Y_prediction

def model(X_train,Y_train,X_test,Y_test,num_ite=2000,learning_rate=0.5,print_cost=True):
	w,b=init(X_train.shape[0])

	params,gards,costs=opt(w,b,X_train,Y_train,num_ite,learning_rate,print_cost=True)

	w=params["w"]
	b=params["b"]

	Y_prediction_test=predict(w,b,X_test)
	Y_prediction_train=predict(w,b,X_train)

	print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
	print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

	d={"cost":costs,"Y_prediction_train":Y_prediction_train,"Y_prediction_test":Y_prediction_test,"w":w,"b":b,"learning_rate":learning_rate,"num_ite":num_ite}

	return d

d=model(train_set_x,train_set_y,test_set_x,test_set_y,num_ite=2000,learning_rate=0.005,print_cost=True)

index = 25
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
plt.show()
print ("y = " + str(test_set_y[0,index]) + ", you predicted that  ")

if d["Y_prediction_test"][0,index]==1:
	print("it's a cat.")
else :
	print("it'not a cat")
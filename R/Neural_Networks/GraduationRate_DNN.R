#Neccessary Libraries
library(neuralnet)
library(ISLR)


#Uploading the data
data <- College

#Getting to know the data
str(data)
dim(data)
summary(data)


#Manipulating the data
max_data <- apply(data[,2:18], 2, max) 
min_data <- apply(data[,2:18], 2, min)

#Scaling the data
data_scaled <- scale(data[,2:18],center = min_data, scale = max_data - min_data) 

#Making some changes
Private = as.numeric(College$Private)-1
data_scaled = cbind(Private,data_scaled)

#70% train
#30% test
index = sample(1:nrow(data),round(0.70*nrow(data)))

#Test/train split
train_data <- as.data.frame(data_scaled[index,])
test_data <- as.data.frame(data_scaled[-index,])




n = names(train_data)
f <- as.formula(paste("Private ~", paste(n[!n %in% "Private"], collapse = " + ")))

#Creating the DNN
deep_net = neuralnet(f,data=train_data,hidden=c(5,3),linear.output=F)

#Visualizing the DNN
plot(deep_net)

predicted_data <- compute(deep_net,test_data[,2:18])
print(head(predicted_data$net.result))
predicted_data$net.result <- sapply(predicted_data$net.result,round,digits=0)

#Making the Confusion Matrix
table(test_data$Private,predicted_data$net.result)





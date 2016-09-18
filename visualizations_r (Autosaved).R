library(ggthemes)
library(ggplot2)
library(reshape2)
library(caret)
library(dplyr)

setwd("~/Dropbox/insight_project/")

product_data <-read.csv("product_info.csv")

f_data <- product_data %>% mutate(price_per_liter_in_usd = price_per_liter_in_cents*0.76/100) %>% select(-price_per_liter_in_cents,-price_per_liter_of_alcohol_in_cents,-price_in_cents,-regular_price_in_cents,-price_per_liter_of_alcohol_in_cents,-tasting_note,-producer_name,-image_url,-description,-is_dead,-tags,-image_thumb_url,-limited_time_offer_savings_in_cents,-serving_suggestion,-limited_time_offer_ends_on, -bonus_reward_miles_ends_on,-updated_at,-released_on,-id,-name,-value_added_promotion_description,-bonus_reward_miles,-sugar_in_grams_per_liter,-tertiary_category,-clearance_sale_savings_in_cents,-varietal,-product_no,-inventory_price_in_cents) %>% mutate(inventory_count=log(inventory_count+1),volume_in_milliliters=log(volume_in_milliliters+1))

f_data$sugar_content<-gsub("^-","",f_data$sugar_content)

f_data<-f_data %>% filter(origin %in% names(which(table(f_data$origin)>50)),package %in% names(which(table(f_data$package)>10)))

dummies <- dummyVars(scrape_id ~ ., data = f_data)
final <- predict(dummies, newdata = f_data)
nzr<-nearZeroVar(final)
f_final<-na.omit(final[,-nzr])
f_final<-as.data.frame(f_final) %>% filter(price_per_liter_in_usd>0)

set.seed(42)
i <-createDataPartition(f_final[,"price_per_liter_in_usd"],p=0.6,list=F)
train <- f_final[i,]
test <- f_final[-i,]

colnames(train)<-make.names(colnames(train))
colnames(test)<-make.names(colnames(test))
ctrl<-trainControl(method="adaptive_cv",repeats=5,number=10,savePredictions=T,index=createResample(train[,"price_per_liter_in_usd"], 50),verbose=T,allowParallel=F,adaptive = list(min = 5,alpha = 0.05,method = "gls",complete = TRUE))
model<-train(price_per_liter_in_usd ~.,data=as.data.frame(train),method="cubist",preProcess=c("range"),trControl=ctrl)

mean(abs((predict(model,train)-train$price_per_liter_in_usd)/train$price_per_liter_in_usd))*100
mean(abs((predict(model,test)-test$price_per_liter_in_usd)/test$price_per_liter_in_usd))*100

data<-read.csv("model_data/performance.csv")
mm<-melt(data,id=c("time_stamp","random_samples"))
ggplot(mm %>% filter(variable !='random'),aes(x=random_samples,y=value,color=variable))+geom_point(fill="white")+geom_line()+theme_bw()+xlab("Randomly Selected Samples")+ylab("Mean Absolute Error (%)")+labs(col="")+ylim(0,100)


png("all.png",width=11,height=6.2,units="in",res=300)
treemap(f_data,
            index=c("primary_category", "secondary_category","name"),
            vSize="inventory_count", vColor="price_per_liter_in_USD",
            type="value",range=c(0,500))
dev.off()

data<-read.csv("model_data/performance.csv")
mm<-melt(data,id=c("time_stamp","random_samples"))
ggplot(mm %>% filter(variable !='random'),aes(x=random_samples,y=value,color=variable,group=variable))+geom_point(fill="white")+geom_smooth()+theme_bw()+xlab("Randomly Selected Samples")+ylab("Mean Absolute Error (%)")+labs(col="")+ylim(0,60)
ggplot(data %>% mutate(performance_vs_chance = random/test),aes(x=random_samples,y= performance_vs_chance))+geom_point(fill="white")+geom_smooth(se=F)+theme_bw()+xlab("Randomly Selected Samples")+ylab("Model performance vs random\n sampling (fold change)")+labs(col="")







pred<-read.csv("check5.csv",row.names=1)
colnames(pred)<-c('label','prediction','actual_price')
pred<-pred %>% mutate(prediction=(2^prediction)*0.76,actual_price=0.76*actual_price,normalize_error=abs(actual_price-prediction)/actual_price,residuals=abs(actual_price-prediction))

ggplot(pred %>% arrange(normalize_error),aes(x=1:dim(pred)[1],y=normalize_error,col=actual_price))+geom_jitter()+theme_bw()+xlab("Index")+ylab("Normalized Error")

ggplot(pred %>% arrange(residuals),aes(x=1:dim(pred)[1],y= log2(residuals),col=actual_price))+geom_jitter()+theme_bw()+xlab("Index")+ylab("residuals")

ggplot(pred ,aes(x=log(residuals),fill="red"))+geom_density()+theme_bw()+xlab("Residuals (log2)")+theme(legend.position="none")
ggplot(pred ,aes(x=log(actual_price),fill="blue"))+geom_density()+theme_bw()+xlab("Actual Price (log2)")+theme(legend.position="none")
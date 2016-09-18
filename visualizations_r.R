png("all.png",width=11,height=6.2,units="in",res=300)
treemap(f_data,
            index=c("primary_category", "secondary_category","name"),
            vSize="inventory_count", vColor="price_per_liter_in_USD",
            type="value",range=c(0,500))
dev.off()

data<-read.csv("learning_curve.csv",row.names=1)
mm<-melt(data,id=c("time_stamp","random_samples"))
ggplot(mm %>% filter(variable !='random'),aes(x=random_samples,y=value,color=variable))+geom_point(fill="white")+geom_line()+theme_bw()+xlab("Randomly Selected Samples")+ylab("Mean Absolute Error (%)")+labs(col="")+ylim(0,100)


pred<-read.csv("check5.csv",row.names=1)
colnames(pred)<-c('label','prediction','actual_price')
pred<-pred %>% mutate(prediction=(2^prediction)*0.76,actual_price=0.76*actual_price,normalize_error=abs(actual_price-prediction)/actual_price,residuals=abs(actual_price-prediction))

ggplot(pred %>% arrange(normalize_error),aes(x=1:dim(pred)[1],y=normalize_error,col=actual_price))+geom_jitter()+theme_bw()+xlab("Index")+ylab("Normalized Error")

ggplot(pred %>% arrange(residuals),aes(x=1:dim(pred)[1],y= log2(residuals),col=actual_price))+geom_jitter()+theme_bw()+xlab("Index")+ylab("residuals")

ggplot(pred ,aes(x=log(residuals),fill="red"))+geom_density()+theme_bw()+xlab("Residuals (log2)")+theme(legend.position="none")
ggplot(pred ,aes(x=log(actual_price),fill="blue"))+geom_density()+theme_bw()+xlab("Actual Price (log2)")+theme(legend.position="none")
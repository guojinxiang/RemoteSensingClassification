A=imread('a.png');
grayJ = rgb2gray(A);
[m,n]=size(grayJ);
width=25;
height=25;
boxes=[];
for y=1:m-height+1
   for x=1:n-width+1
       B=im2double(grayJ(y:y+height-1,x:x+width-1));
       probability=myNeuralNetworkFunction(reshape(B',1,[]));
       if probability>0.9
           r=[x y x+width-1 y+height-1 probability];
           boxes=[boxes;r]; 
       end
   end
end

pick = nms(boxes, 0.0);
for (i=1:size(pick,1))
    [state, A] =draw_rect(A,[boxes(pick(i),1),boxes(pick(i),2),boxes(pick(i),3)-boxes(pick(i),1),boxes(pick(i),4)-boxes(pick(i),2)],0); 
end  
imwrite(A,['result/output.jpg']);
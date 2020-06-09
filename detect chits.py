import cv2
import numpy as np   
import tensorflow as tf
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


font = cv2.FONT_HERSHEY_SIMPLEX 
# Read the graph.
with tf.gfile.FastGFile('mobilenet/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')


    for k in range(1,30):

        # Read and preprocess an image.
        img = cv2.imread('mobilenet/new2/t ({}).jpg'.format(k))
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv2.resize(img, (300, 300))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                    feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
        
        
        
        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        person=False
        for i in range(num_detections):
            
            
            
            classId = int(out[3][0][i])
        
            if classId!=1:
                
                continue
        
            else:
                person=True
                score = float(out[1][0][i])
                bbox = [float(v) for v in out[2][0][i]]
                if score > 0.3:
                    x = bbox[1] * cols
                    y = bbox[0] * rows
                    right = bbox[3] * cols
                    bottom = bbox[2] * rows
                    cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)

        if person==True: 
            img = cv2.putText(img,'Person is present',(100,100), font, 2,(169, 245, 159),4,cv2.LINE_AA)

            cv2.imshow('TensorFlow MobileNet-SSD', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            img = cv2.imread('mobilenet/new2/t ({}).jpg'.format(k))
            image = cv2.GaussianBlur(img,(5,5),0)
            
            
            
        # img = image[249:375,202:394]
            img = image.copy()

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            dilation = cv2.dilate(img,kernel,iterations = 3)

            dilation = cv2.dilate(dilation, None, iterations=1)
            dilation = cv2.erode(dilation, None, iterations=1)



            img1 = dilation.copy()
            for i in range(np.shape(dilation)[0]):
                for j in range(np.shape(dilation)[1]):
                    b,g,r = dilation[i][j]

                    l= 210
                    u= 255

                    if l<b<u and l<g<u and l<r<u:
                        b,g,r      = 255,255,255
                        img1[i][j] =  b,g,r
                    else :
                        b,g,r      = 0,0,0
                        img1[i][j] =  b,g,r




            img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            contours, hierarchy = cv2.findContours(img2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            pointer=0
            for i in range(len(contours)):
                
                M = cv2.moments(contours[i])
                cX = int(M["m10"] / M["m00"]) 
                #cX = cX + 249
                cY = int(M["m01"] / M["m00"])
                #cY = cY + 202
                print(cX,cY)
                point = Point(cX, cY)
                polygon = Polygon([(84,198),(142,401),(208,158),(395,401)])
                inside = (polygon.contains(point))
                print(inside)
                
                if inside == True:
                    a=cv2.contourArea(contours[i])
                    print(a)
                    if a>0 and a<800:
                            cv2.drawContours(img, contours, i, (0,255,0), 1)
                            cv2.drawContours(dilation, contours, i, (0,255,0), 1)
                            pointer+=1
                else:
                    continue

            font = cv2.FONT_HERSHEY_SIMPLEX 
            if pointer>2:
                image = cv2.putText(img,'May not be clean',(100,100), font, 1,(12, 12, 232),4,cv2.LINE_AA)
            else:
                image = cv2.putText(img,'may be clean',(100,100), font, 1,(169, 245, 159),4,cv2.LINE_AA)



            
            #cv2.imshow('image',image)
            cv2.imshow('cropped',img)
            cv2.imshow('threshold', img1)
            cv2.imshow('dilation', dilation)
            cv2.waitKey(0)
            cv2.destroyAllWindows()    
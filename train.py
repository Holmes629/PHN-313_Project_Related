import face_recognition
import numpy as np
import cv2
import os 
from pickle import load,dump
from time import time

k=input('(a) add a user \n(b) Delete a user \n')

while k=='a':
    name=input('Please input you name: ')               # Asking a name o save the data
    print('Please look at the camera for 5 seconds :)') # photo for face encoding

    path=os.path.dirname(__file__)+'\Photos'            #path as Photos in our dir; If not present create a directory 
    if os.path.exists(path)==False:
        os.mkdir(path)

    cam= cv2.VideoCapture(0)                            #capturing the image

    start_t=time(); end_t=time()
   
    while (end_t-start_t)<=5:
        result, image=cam.read()                        #reading the image
        cv2.imshow('image',image)
        end_t=time()
        if cv2.waitKey(1) & 0xFF == ord('q'):           # exit if typed alphabet q
            break
    
    cam.release()
    cv2.destroyAllWindows()                             # closing the windows of capturng picture
    
    if result:                                          #if result =1 then store the image to pgotos folder with their name.jpg
        cv2.imwrite(os.path.join(path,name+'.jpg'),image) 
        print('Sucessfully added '+name+' to the database')
        
        image=os.path.join(path,name+'.jpg')            # image path for software to read and encode the facial data
        dataset=os.path.dirname(__file__)+'/dataset_faces.dat'  # creating a file 'dataset_faes.dat' for storing the facial data

        if os.path.exists(dataset)==False:              # if dataset file is not avialble then create dataset
            data=open(dataset,'w')
            data.close()

        if os.stat(dataset).st_size==0:                 # if the dataset file is empty then append {} to the data as python can't read a empty file
            with open(dataset,'wb') as data:
                dump({},data)

        with open(dataset, 'rb') as data:               #opening dataset file to read the data
            all_face_encodings = load(data)      # loading the previoud data to append the new face data

        img=face_recognition.load_image_file(image)     # loading the image by using module which mload an image RGB values to numpy array
        all_face_encodings[name] = face_recognition.face_encodings(img)[0]  # convert image to HOG and create a 128 dataset values for facial recognition and adding it to dictionary with their name

        with open(dataset, 'wb') as data:               # opening dataset.dat to write new facial data
            dump(all_face_encodings, data)
        data.close()                                    # closing the file to avoid errors
        k=0
    else:
        print('No image detected, Please try again')        #if no image detected return this error

    

if k=='b':                                                    #for deleting a user
    dataset=os.path.dirname(__file__)+'/dataset_faces.dat'

    with open(dataset, 'rb') as data:                         # opening the dataset.dat file
        all_face_encodings = load(data)

        if len(list(all_face_encodings.keys()))==0:             # if no user data present return error
            print('\n NO USER PRESENT')
        else:
            print(list(all_face_encodings.keys()))                # list all the names in the dataset.dat file here "keys=names"
            print('Select an user from above')

            n=input('please input name of the user: ')
            if n not in all_face_encodings:                          # if name not in the list then print an error
                print('User is not present/ Already deleted')
            else:                                                   # else delete the user and his facial data
                del all_face_encodings[n]
                print('User is deleted')

            with open(dataset, 'wb') as data:                       # write the new data( with removed facial data of requested user to file)
                dump(all_face_encodings, data)
            data.close()

    





    











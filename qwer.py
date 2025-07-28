import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
import pywt
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class DicomViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BMP Image Viewer")
        self.canvas = tk.Canvas(self, width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image_item = None
        self.original_image = None
        self.current_image_index = 0
        self.image_files = []
        self.folder_path = None
        self.feat_list = []  # Список для хранения массивов feat
        self.image_count = 0  # Счетчик обработанных изображений

        open_button = tk.Button(self, text="Open BMP Folder", command=self.open_bmp_folder)
        open_button.pack(side=tk.TOP, pady=10)

        next_button = tk.Button(self, text="Next Image", command=self.load_next_image)
        next_button.pack(side=tk.TOP, pady=10)

        process_all_button = tk.Button(self, text="Process All Images", command=self.process_all_images)
        process_all_button.pack(side=tk.TOP, pady=10)
        
        open_excel_button = tk.Button(self, text="Open Excel File", command=self.open_excel_file)
        open_excel_button.pack(side=tk.TOP, pady=10)

    def open_bmp_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.folder_path = folder_path
            self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.bmp')]
            if self.image_files:
                self.current_image_index = 0
                self.load_and_process_bmp(os.path.join(folder_path, self.image_files[self.current_image_index]))

    def load_next_image(self):
        if self.current_image_index + 1 < len(self.image_files):
            self.current_image_index += 1
            next_file_path = os.path.join(self.folder_path, self.image_files[self.current_image_index])
            self.load_and_process_bmp(next_file_path)
        else:
            messagebox.showinfo("Info", "This was the last image in the folder.")

    def process_all_images(self):
        if not self.image_files:
            messagebox.showinfo("Info", "No BMP files found in the folder.")
            return

        for i, image_file in enumerate(self.image_files):
            file_path = os.path.join(self.folder_path, image_file)
            self.load_and_process_bmp(file_path)
            self.current_image_index = i
            if i < len(self.image_files) - 1:
                continue
            else:
                messagebox.showinfo("Info", "This was the last image in the folder.")
                
        self.save_feat_to_excel()
        
    def save_feat_to_excel(self):
        if not self.feat_list:
            messagebox.showinfo("Info", "No features to save.")
            return

        # Получаем родительскую директорию для пути к папке с изображениями
        parent_directory = os.path.dirname(self.folder_path)
        excel_file_path = os.path.join(parent_directory, 'features.xlsx')

        df = pd.DataFrame(self.feat_list, columns=[f'Feat{i}' for i in range(62)])  # Теперь у нас 62 признака
        df.to_excel(excel_file_path, index=False)
        messagebox.showinfo("Info", f"Features saved to {excel_file_path}")

    def load_and_process_bmp(self, file_path):
        try:
            with Image.open(file_path) as img:
                self.original_image = img
                self.update_image_size()
                self.process_image(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {e}")

    def process_image(self, file_path):
        image_array = np.array(self.original_image.convert('L'))
        print(f"Processed image from file: {file_path}")
        print(f"Image dimensions: {image_array.shape}")
        feat = [0]*61
        hist_im = np.zeros(256)
        for i in range (image_array.shape[0]):
            for j in range (image_array.shape[1]):
                hist_im[image_array[i][j]]+=1
        hist_im /= (image_array.shape[0] * image_array.shape[1])
        feat[0] = self.getMean(hist_im) #hist_mean
        feat[1] = self.getVariance(hist_im,feat[0]) #hist_var
        feat[2] = self.getSkewness(hist_im,feat[0],feat[1]) #hist_skew
        feat[3] = self.getKurtosis(hist_im,feat[0],feat[1]) #hist_kurt
        feat[4] = self.getPerc(hist_im,0.01) #perc01
        feat[5] = self.getPerc(hist_im,0.1) #perc10
        feat[6] = self.getPerc(hist_im,0.5) #perc50
        #feat[7] = self.getPerc(hist_im,0.9) #perc90
        #feat[8] = self.getPerc(hist_im,0.99) #perc99
        CCMP = np.zeros((256,256))
        Px = np.zeros(256)
        Py = np.zeros(256)
        PxPy=np.zeros(256*2)
        PxMy=np.zeros(256*2)
        #feat[9:20] = self.getHaralickFeatures(image_array,CCMP,Px,Py,PxPy,PxMy,1,0) #ASM,SofS,SA,SV,SE,IDM,Cont,Corr, Ent, Dv, DE
        #feat[20:31] = self.getHaralickFeatures(image_array,CCMP,Px,Py,PxPy,PxMy,0,1)
        feat[7:18] = self.getHaralickFeatures(image_array,CCMP,Px,Py,PxPy,PxMy,1,1)
        #feat[42:53] = self.getHaralickFeatures(image_array,CCMP,Px,Py,PxPy,PxMy,1,-1)
        #feat[53:64] = self.getHaralickFeatures(image_array,CCMP,Px,Py,PxPy,PxMy,2,0)
        #feat[64:75] = self.getHaralickFeatures(image_array,CCMP,Px,Py,PxPy,PxMy,0,2)
        #feat[75:86] = self.getHaralickFeatures(image_array,CCMP,Px,Py,PxPy,PxMy,2,2)
        #feat[86:97] = self.getHaralickFeatures(image_array,CCMP,Px,Py,PxPy,PxMy,2,-2)
        feat[18:29] = self.getHaralickFeatures(image_array,CCMP,Px,Py,PxPy,PxMy,3,0)
        #feat[108:119] = self.getHaralickFeatures(image_array,CCMP,Px,Py,PxPy,PxMy,0,3)
        #feat[119:130] = self.getHaralickFeatures(image_array,CCMP,Px,Py,PxPy,PxMy,3,3)
        #feat[130:141] = self.getHaralickFeatures(image_array,CCMP,Px,Py,PxPy,PxMy,3,-3)
        #feat[141:152] = self.getHaralickFeatures(image_array,CCMP,Px,Py,PxPy,PxMy,4,0)
        #feat[152:163] = self.getHaralickFeatures(image_array,CCMP,Px,Py,PxPy,PxMy,0,4)
        feat[29:40] = self.getHaralickFeatures(image_array,CCMP,Px,Py,PxPy,PxMy,4,4)
        #feat[174:185] = self.getHaralickFeatures(image_array,CCMP,Px,Py,PxPy,PxMy,4,-4)
        #feat[185:196] = self.getHaralickFeatures(image_array,CCMP,Px,Py,PxPy,PxMy,5,0)
        #feat[196:207] = self.getHaralickFeatures(image_array,CCMP,Px,Py,PxPy,PxMy,0,5)
        #feat[207:218] = self.getHaralickFeatures(image_array,CCMP,Px,Py,PxPy,PxMy,5,5)
        feat[40:51] = self.getHaralickFeatures(image_array,CCMP,Px,Py,PxPy,PxMy,5,-5)
        rowsH = 256
        colsH = image_array.shape[1]
        matH = np.zeros((rowsH,colsH))
        feat[51:56]=self.CalcRunLengthFeaturesHight(image_array,matH) #HorzlShrtREmph,HorzlLngREmphself,HorzlGLevNonUni,HorzlRLNonUni,HorzlFraction
        rowsW = 256
        colsW = image_array.shape[0]
        matW = np.zeros((rowsW,colsW))
        #feat[234:239]=self.CalcRunLengthFeaturesW(image_array,matW) #VertlShrtREmph,VertlLngREmphself,VertlGLevNonUni,VertlRLNonUni,VertlFraction
        grad_t = self.getAbsoluteGradient(image_array)
        feat[56:61] = self.getGradientFeatures(grad_t)
        
        if self.image_count < 100:
            new_feature = 0
        else:
            new_feature = 1
        feat.insert(0, new_feature)
                
        self.feat_list.append(feat)
        self.image_count += 1
        # Format the statistics into a table
        table_text = f"Statistics for the selected area:\n\n"        
        #table_text += f"S(1,0)SumOfSqs: {SofS10}\n"
        table_text += f"S(1,1)SumVarnc: {feat[13]}\n"
        table_text += f"S(3,0)SumEntrp: {feat[25]}\n"
        table_text += f"S(4,4)SumAverg: {feat[34]}\n"
        table_text += f"S(5,-5)SumOfSqs: {feat[43]}\n"
        #table_text += f"VertlGLevNonUni: {VertlGLevNonUni}\n"
        #table_text += f"WavEnLH_s-3: {WavEnLH_s3}\n"
        #table_text += f"S(2,-2)DifEntrp: {DE2min2}\n"
        #table_text += f"S(5,0)DifEntrp: {IDM50}\n"
        #table_text += f"WavEnHH_s-4: {WavEnHH_s4}\n"
        #table_text += f"WavEnLH_s-2: {WavEnLH_s2}\n"
        #table_text += f"S(0,3)SumVarnc: {SV03}\n"
        table_text += f"HorzlRLNonUni: {feat[51]}\n"
        table_text += f"Variance: {feat[1]}\n"
        table_text += f"Skewness: {feat[3]}\n"
        # Show the table in a message box
        messagebox.showinfo("Pixel Statistics", table_text)


    def getMean(self,hist):
        mean = 0
        for i in range (hist.shape[0]):
            mean+=(i+1)*hist[i]
        return mean
    
    
    def getVariance(self,hist,mean):
        var = 0
        for i in range (hist.shape[0]):
            var+=pow((i+1)-mean,2)*hist[i]
        return var
    
    
    def getSkewness(self,hist,mean,var):
        skew = 0
        for i in range (hist.shape[0]):
            skew+=pow((i+1)-mean,3)*hist[i]
        return skew/pow(var,1.5)    
    
    
    def getKurtosis(self,hist,mean,var):
        kurt = 0
        for i in range (hist.shape[0]):
            kurt+=pow((i+1)-mean,4)*hist[i]
        return (kurt/pow(var,2)-3)
    
    
    def getPerc(self,hist,val):
        t =0
        perc = 0
        while t < val:
            t +=hist[perc]
            perc+=1
        return perc
    
    
    def CoocurateMatrix(self,array,CCMP,dx,dy):
        temp = np.zeros((256,256))
        sum=0
        for i in range (abs(min(0,dy)),(array.shape[0]-max(0,dy))):
            for j in range (abs(min(0,dx)),(array.shape[1]-max(0,dx))):
                k=(array[i][j])
                l=(array[i+dy][j+dx])
                k=int(k/1)
                l=int(l/1)
                temp[k][l]+=1
                temp[l][k]+=1
                sum+=1
        for i in range (256):
            for j in range (256):
                CCMP[i][j]=float(temp[i][j]/(2*sum))
        return CCMP
    
    
    def getAngularSecondMoment(self,CCMP):
        res=0
        for i in range (256):
            for j in range (256):
                res+=pow(CCMP[i][j],2)
        return res
    
    
    def Calc_Px(self,CCMP):
        t =np.zeros(256)
        for i in range (256):
            for j in range (256):
                t[j]+=CCMP[i][j]
        return t
    
    
    def Calc_Py(self,CCMP):
        t = np.zeros(256)
        for i in range (256):
            for j in range (256):
                t[i]+=CCMP[i][j]        
        return t
        
    
    def Calc_M(self,P):
        t = 0
        for i in range (1,256):
            t+=i*P[i]
        return t
    
    
    def Calc_D(self,P,M):
        t = 0
        for i in range (256):
            t+= (pow(i-M,2)*P[i])
        return t
    
    
    def GetSumOfSquares(self,CCMP,Mx):
        res=0
        for i in range (256):
            for j in range (256):
                di=i-Mx
                res+=di*di*CCMP[i][j]
        return res
    
    
    def Calc_PxPy(self,CCMP):
        t =np.zeros(256*2)
        for i in range (256):
            for j in range (256):
                t[i+j]+=CCMP[i][j]
        return t
    
    
    def Calc_PxMy(self,CCMP):
        t =np.zeros(256*2)
        for i in range (256):
            for j in range (256):
                di = abs(i-j)
                t[di]+=CCMP[i][j]
        return t
    
    
    def GetSumAverage(self,PxPy):
        t = 0
        for i in range (256*2):
            if (PxPy[i]>0):
                t+=i*PxPy[i]
        return t
    
    
    def GetSumVariance(self,PxPy,SA):
        t = 0
        for i in range (0,256*2):
            if (PxPy[i]>0):
                t += pow((i+1-SA),2)*PxPy[i]
        return t
    
    
    def GetSumEntropy(self,PxPy):
        t = 0
        for i in range (0,256*2):
            if (PxPy[i]>0):
                t += PxPy[i]*math.log10(PxPy[i])
        return -t
    
    
    def GetInverseDifferenceMoment(self,CCMP):
        t = 0
        for i in range(256):
            for j in range(256):
                di = i-j
                t+=CCMP[i][j]/(1+di*di)
        return t
    
    
    def getContrast(self,CCMP):
        t = 0
        for i in range(256):
            for j in range(256):
                t+=(i-j)*(i-j)*CCMP[i][j]
        return t
        
    
    def getCorr(self,CCMP,Mx,My,Dx,Dy):
        t = 0
        for i in range(256):
            for j in range(256):
                t+=(i*j)*CCMP[i][j]
        t-=(Mx*My)
        t/=(Dx*Dy)
        return t
    
    
    def  GetEntropy(self,CCMP):
        t = 0
        for i in range(256):
            for j in range(256):
                if (CCMP[i][j]>0):
                    t+=CCMP[i][j]*math.log10(CCMP[i][j])
        return -t        
        
    
    def GetDiffereceVariance(self,PxMy,SE):
        t = 0
        for i in range (0,256*2):
            if (PxMy[i]>0):
                t += pow ((i-SE),2)*PxMy[i]
        return t
    
    
    def GetDefferenceEntropy(self,PxMy):
        t = 0
        for i in range(256*2):
            if (PxMy[i]>0):
                t+=PxMy[i]*math.log10(PxMy[i])
        return -t
    
          
    def getHaralickFeatures(self, array,CCMP,Px,Py,PxPy,PxMy,dx,dy):
        CCMP = self.CoocurateMatrix(array,CCMP,dx,dy)
        Asm = self.getAngularSecondMoment(CCMP)
        Px = self.Calc_Px(CCMP)
        Py =self.Calc_Py(CCMP)
        Mx = self.Calc_M(Px)
        My = self.Calc_M(Py)
        Dx = self.Calc_D(Px, Mx)
        Dy = self.Calc_D(Py, My)
        SofS = self.GetSumOfSquares(CCMP, Mx)
        PxPy = self.Calc_PxPy(CCMP)
        PxMy = self.Calc_PxMy(CCMP)
        SA = self.GetSumAverage(PxPy)
        SV = self.GetSumVariance(PxPy,SA)
        SE = self.GetSumEntropy(PxPy)
        IDM = self.GetInverseDifferenceMoment(CCMP)
        Cont = self.getContrast(CCMP)
        Corr = self.getCorr(CCMP,Mx,My,Dx,Dy)
        Ent = self.GetEntropy(CCMP)
        Dv = self.GetDiffereceVariance(PxMy,SE)
        De = self.GetDefferenceEntropy(PxMy)
        return Asm, SofS, SA, SV, SE, IDM,Cont,Corr, Ent, Dv, De
    
    def calcRunLengthMatrixHight(self,im,arr):
        di=0
        dj=1
        t0=0
        t1=im.shape[0]
        count=im.shape[1]
        for t in range(t0,t1):
            posI=t
            posJ=0
            runlen=0
            val=im[posI][posJ]
            lastValue = val
            runlen+=1
            posI+=di
            posJ+=dj
            for c in range(0,count-1):
                val=im[posI][posJ]
                if lastValue!=val:
                    arr[lastValue][runlen-1]+=1
                    runlen = 0
                lastValue = val
                runlen+=1
                posI+=di
                posJ+=dj
            arr[lastValue][runlen-1]+=1
        return arr
    

    def calcShrtREmph(self,arr,C):
        res = 0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                res+=float(arr[i][j])/float((j+1)*(j+1))
        return res/C
    
    
    def calcLngREmph(self,arr,C):
        res = 0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                res+=float(arr[i][j])*((j+1)*(j+1))
        return res/C    
    
    def calcGLevNonUni(self,arr,C):
        res=0
        locres=0
        for i in range(0,arr.shape[0]):
            locres=0
            for j in range(0,arr.shape[1]):
                locres+=(arr[i][j])
            res+=pow(locres,2)
        return (res/C)
    
    def calcRLNonUni(self,arr,C):
        res=0
        locres=0
        for j in range(0,arr.shape[1]):
            locres=0
            for i in range(0,arr.shape[0]):
                locres+=(arr[i][j])
            res+=pow(locres,2)
        return (res/C)
   
    def calcFraction(self,arr,C):
        res=0
        for i in range(0,arr.shape[0]):
            for j in range(0,arr.shape[1]):
                res+=(arr[i][j])*(j+1)
        return (C/res)
 
    def CalcRunLengthFeaturesHight(self,im,arr):
        arr = self.calcRunLengthMatrixHight(im,arr)
        C=np.sum(arr)
        ShrtREmph = self.calcShrtREmph(arr,C)
        LngREmphself = self.calcLngREmph(arr,C)
        GLevNonUni = self.calcGLevNonUni(arr,C)
        RLNonUni = self.calcRLNonUni(arr,C)
        Fraction= self.calcFraction(arr,C)
        return ShrtREmph, LngREmphself, GLevNonUni, RLNonUni, Fraction
    
    def calcRunLengthMatrixW(self,im,arr):
        di=1
        dj=0
        t0=0
        t1=im.shape[1]
        count=im.shape[0]
        for t in range(t0,t1):
            posI=0
            posJ=t
            runlen=0
            val=im[posI][posJ]
            lastValue = val
            runlen+=1
            posI+=di
            posJ+=dj
            for c in range(0,count-1):
                val=im[posI][posJ]
                if lastValue!=val:
                    arr[lastValue][runlen-1]+=1
                    runlen = 0
                lastValue = val
                runlen+=1
                posI+=di
                posJ+=dj
            arr[lastValue][runlen-1]+=1
        return arr
    
    def CalcRunLengthFeaturesW(self,im,arr):
        arr = self.calcRunLengthMatrixW(im,arr)
        C=np.sum(arr)
        ShrtREmph = self.calcShrtREmph(arr,C)
        LngREmphself = self.calcLngREmph(arr,C)
        GLevNonUni = self.calcGLevNonUni(arr,C)
        RLNonUni = self.calcRLNonUni(arr,C)
        Fraction= self.calcFraction(arr,C)
        return ShrtREmph, LngREmphself, GLevNonUni, RLNonUni, Fraction

    def update_image_size(self):
        if self.original_image is None:
            return
        original_width, original_height = self.original_image.size
        self.canvas.config(width=original_width, height=original_height)
        resized_image = self.original_image.resize((original_width, original_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized_image)
        if self.image_item:
            self.canvas.delete(self.image_item)
            self.image_item = self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        else:
            self.image_item = self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            
    def getAbsoluteGradient(self,im):
        gradRows=im.shape[0]-2
        gradCols=im.shape[1]-2
        res = np.zeros((gradRows, gradCols))
        for a in range(1,im.shape[0]-1):
            for j in range(1,im.shape[1]-1):
                di=int(im[a+1][j])-int(im[a-1][j])
                dj=int(im[a][j+1])-int(im[a][j-1])
                res[a-1][j-1]=math.sqrt(di*di+dj*dj)
        return res
    
    def getGradientFeatures(self,grad):
        absgrmean=self.GetAbsoluteGradientMean(grad)
        absgrsigma = self.GetAbsoluteGradientSigma(grad,absgrmean)
        absgrkurt = self.GetAbsoluteGradientKurtosis(grad,absgrmean,absgrsigma)
        absgrskwn = self.GetAbsoluteGradientSkewness(grad,absgrmean,absgrsigma)
        absgrnonz = self.GetAbsoluteGradientNonZeros(grad)
        return absgrmean, absgrsigma, absgrkurt, absgrskwn, absgrnonz
        
    
    def GetAbsoluteGradientMean(self,grad):
        res=0
        for i in range(1,grad.shape[0]):
            for j in range(1,grad.shape[1]):
                res+=grad[i][j]
        return res/((grad.shape[0])*(grad.shape[1]))
    
    def GetAbsoluteGradientSigma(self,grad,absgrmean):
        res=0
        for i in range(1,grad.shape[0]):
            for j in range(1,grad.shape[1]):
                res+=pow(grad[i][j]-absgrmean,2)
        return res/((grad.shape[0])*(grad.shape[1]))
    
    def GetAbsoluteGradientKurtosis(self,grad,absgrmean,absgrsigma):
        res=0
        for i in range(1,grad.shape[0]):
            for j in range(1,grad.shape[1]):
                res+=pow(grad[i][j]-absgrmean,4)
        return ((res/(grad.shape[0]*grad.shape[1]*pow(absgrsigma,2)))-3)
    
    def GetAbsoluteGradientSkewness(self,grad,absgrmean,absgrsigma):
        res=0
        for i in range(1,grad.shape[0]):
            for j in range(1,grad.shape[1]):
                res+=pow(grad[i][j]-absgrmean,3)
        return (res/(grad.shape[0]*grad.shape[1]*pow(absgrsigma,1.5)))
    
    def GetAbsoluteGradientNonZeros(self,grad):
        res=0
        for i in range(1,grad.shape[0]):
            for j in range(1,grad.shape[1]):
                if (grad[i][j]>0):
                    res+=1
        return (res/(grad.shape[0]*grad.shape[1]))
    
    
    def open_excel_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            self.process_excel_data(file_path)
            
            
    def process_excel_data(self, file_path):
        try:
            result_text = "Точность выявления заболевания остеопорозом: \n\n"
            df = pd.read_excel(file_path, header=None)
            y = df.iloc[:, 0]
            X = df.iloc[:, 1:]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train1 = X_train.iloc[:, 0:7]
            X_test1 = X_test.iloc[:, 0:7]
            X_train_scaled = X_train1
            X_test_scaled = X_test1
            lda = LinearDiscriminantAnalysis(n_components=1)
            X_train_lda = lda.fit_transform(X_train_scaled, y_train)
            X_test_lda = lda.transform(X_test_scaled)
            X_train_new = pd.concat([pd.DataFrame(X_train1), pd.DataFrame(X_train_lda, index=X_train1.index)], axis=1)
            X_test_new = pd.concat([pd.DataFrame(X_test1), pd.DataFrame(X_test_lda, index=X_test1.index)], axis=1)
            model_with_lda = RandomForestClassifier()
            model_with_lda.fit(X_train_new, y_train)
            y_pred_with_lda = model_with_lda.predict(X_test_new)
            accuracy_with_lda_1 = accuracy_score(y_test, y_pred_with_lda)
            result_text += f"Histogram features: {accuracy_with_lda_1:.8f}\n\n"
 


            X_train1 = X_train.iloc[:, 7:18]
            X_test1 = X_test.iloc[:, 7:18]
            X_train_scaled = X_train1
            X_test_scaled = X_test1
            lda = LinearDiscriminantAnalysis(n_components=1)
            X_train_lda = lda.fit_transform(X_train_scaled, y_train)
            X_test_lda = lda.transform(X_test_scaled)
            X_train_new = pd.concat([pd.DataFrame(X_train1), pd.DataFrame(X_train_lda, index=X_train1.index)], axis=1)
            X_test_new = pd.concat([pd.DataFrame(X_test1), pd.DataFrame(X_test_lda, index=X_test1.index)], axis=1)
            model_with_lda = RandomForestClassifier()
            model_with_lda.fit(X_train_new, y_train)
            y_pred_with_lda = model_with_lda.predict(X_test_new)
            accuracy_with_lda_2 = accuracy_score(y_test, y_pred_with_lda)
            result_text += f"Features of Haralik(1,1): {accuracy_with_lda_2:.8f}\n\n"


            X_train1 = X_train.iloc[:, 18:29]
            X_test1 = X_test.iloc[:, 18:29]
            X_train_scaled = X_train1
            X_test_scaled = X_test1
            lda = LinearDiscriminantAnalysis(n_components=1)
            X_train_lda = lda.fit_transform(X_train_scaled, y_train)
            X_test_lda = lda.transform(X_test_scaled)
            X_train_new = pd.concat([pd.DataFrame(X_train1), pd.DataFrame(X_train_lda, index=X_train1.index)], axis=1)
            X_test_new = pd.concat([pd.DataFrame(X_test1), pd.DataFrame(X_test_lda, index=X_test1.index)], axis=1)
            model_with_lda = RandomForestClassifier()
            model_with_lda.fit(X_train_new, y_train)
            y_pred_with_lda = model_with_lda.predict(X_test_new)
            accuracy_with_lda_3 = accuracy_score(y_test, y_pred_with_lda)
            result_text+= f"Features of Haralik(3,0): {accuracy_with_lda_3:.8f}\n\n"
            

            X_train1 = X_train.iloc[:, 29:40]
            X_test1 = X_test.iloc[:, 29:40]
            X_train_scaled = X_train1
            X_test_scaled = X_test1
            lda = LinearDiscriminantAnalysis(n_components=1)
            X_train_lda = lda.fit_transform(X_train_scaled, y_train)
            X_test_lda = lda.transform(X_test_scaled)
            X_train_new = pd.concat([pd.DataFrame(X_train1), pd.DataFrame(X_train_lda, index=X_train1.index)], axis=1)
            X_test_new = pd.concat([pd.DataFrame(X_test1), pd.DataFrame(X_test_lda, index=X_test1.index)], axis=1)
            model_with_lda = RandomForestClassifier()
            model_with_lda.fit(X_train_new, y_train)
            y_pred_with_lda = model_with_lda.predict(X_test_new)
            accuracy_with_lda_4 = accuracy_score(y_test, y_pred_with_lda)
            result_text += f"Features of Haralik(4,4): {accuracy_with_lda_4:.8f}\n\n"
            

            X_train1 = X_train.iloc[:, 40:51]
            X_test1 = X_test.iloc[:, 40:51]
            X_train_scaled = X_train1
            X_test_scaled = X_test1
            lda = LinearDiscriminantAnalysis(n_components=1)
            X_train_lda = lda.fit_transform(X_train_scaled, y_train)
            X_test_lda = lda.transform(X_test_scaled)
            X_train_new = pd.concat([pd.DataFrame(X_train1), pd.DataFrame(X_train_lda, index=X_train1.index)], axis=1)
            X_test_new = pd.concat([pd.DataFrame(X_test1), pd.DataFrame(X_test_lda, index=X_test1.index)], axis=1)
            model_with_lda = RandomForestClassifier()
            model_with_lda.fit(X_train_new, y_train)
            y_pred_with_lda = model_with_lda.predict(X_test_new)
            accuracy_with_lda_5 = accuracy_score(y_test, y_pred_with_lda)
            result_text += f"Features of Haralik(5,-5): {accuracy_with_lda_5:.8f}\n\n"

            
            

            X_train1 = X_train.iloc[:, 51:56]
            X_test1 = X_test.iloc[:, 51:56]
            X_train_scaled = X_train1
            X_test_scaled = X_test1
            lda = LinearDiscriminantAnalysis(n_components=1)
            X_train_lda = lda.fit_transform(X_train_scaled, y_train)
            X_test_lda = lda.transform(X_test_scaled)
            X_train_new = pd.concat([pd.DataFrame(X_train1), pd.DataFrame(X_train_lda, index=X_train1.index)], axis=1)
            X_test_new = pd.concat([pd.DataFrame(X_test1), pd.DataFrame(X_test_lda, index=X_test1.index)], axis=1)
            model_with_lda = RandomForestClassifier()
            model_with_lda.fit(X_train_new, y_train)
            y_pred_with_lda = model_with_lda.predict(X_test_new)
            accuracy_with_lda_6 = accuracy_score(y_test, y_pred_with_lda)
            result_text += f"Features based on group matrice: {accuracy_with_lda_6:.8f}\n\n"

            
            

            X_train1 = X_train.iloc[:, 56:61]
            X_test1 = X_test.iloc[:, 56:61]
            X_train_scaled = X_train1
            X_test_scaled = X_test1
            lda = LinearDiscriminantAnalysis(n_components=1)
            X_train_lda = lda.fit_transform(X_train_scaled, y_train)
            X_test_lda = lda.transform(X_test_scaled)
            X_train_new = pd.concat([pd.DataFrame(X_train1), pd.DataFrame(X_train_lda, index=X_train1.index)], axis=1)
            X_test_new = pd.concat([pd.DataFrame(X_test1), pd.DataFrame(X_test_lda, index=X_test1.index)], axis=1)
            model_with_lda = RandomForestClassifier()
            model_with_lda.fit(X_train_new, y_train)
            y_pred_with_lda = model_with_lda.predict(X_test_new)
            accuracy_with_lda_7 = accuracy_score(y_test, y_pred_with_lda)
            result_text += f"Gradient features: {accuracy_with_lda_7:.8f}\n\n"
            
            messagebox.showinfo("Discriminant analisys", result_text)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось обработать данные: {e}")

if __name__ == "__main__":
    app = DicomViewer()
    app.mainloop()
"""Synthetic native-format source collections, deliberately unrelated to clinical images."""
from pathlib import Path

import numpy as np
import pandas as pd
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import DigitalMammographyXRayImageStorageForPresentation, ExplicitVRLittleEndian


def projection(path: Path, patient: str, n: int, view="CC", mono="MONOCHROME2"):
    path.parent.mkdir(parents=True,exist_ok=True)
    meta=FileMetaDataset()
    meta.MediaStorageSOPClassUID=DigitalMammographyXRayImageStorageForPresentation
    meta.MediaStorageSOPInstanceUID=f"2.25.{n+1}"
    meta.TransferSyntaxUID=ExplicitVRLittleEndian
    ds=FileDataset(str(path),{},file_meta=meta,preamble=bytes(128))
    ds.SOPClassUID=meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID=f"2.25.{n+10000}"
    ds.SeriesInstanceUID=f"2.25.{n+20000}"
    ds.PatientID=patient
    ds.ImageLaterality="L"
    ds.ViewPosition=view
    ds.Rows,ds.Columns=16,8
    ds.BitsAllocated,ds.BitsStored,ds.HighBit=16,12,11
    ds.SamplesPerPixel,ds.PixelRepresentation=1,0
    ds.PhotometricInterpretation=mono
    ds.PixelData=(np.random.default_rng(n).integers(1,4095,size=(16,8),dtype="<u2")).tobytes()
    ds.save_as(path,enforce_file_format=True)
    return ds


def make_sources(root: Path):
    configs={}
    for i,name in enumerate(("CBIS-DDSM","CMMD","INbreast")):
        source=root/name
        rows=[]
        for j in range(8):
            patient=f"P_{j:05d}"
            file=f"{j}_scan.dcm"
            projection(source/file,patient,1000*i+j)
            if i==0:
                rows.append(dict(patient_id=patient,**{"left or right breast":"LEFT","image view":"CC","abnormality id":"1",
                    "abnormality type":"mass","image file path":file,"cropped image file path":f"crop{j}","ROI mask file path":f"mask{j}",
                    "breast_density":str(1+j%4),"pathology":["BENIGN","MALIGNANT"][j%2]}))
            if i==1:
                rows.append(dict(ID1=patient,LeftRight="L",classification=["Benign","Malignant"][j%2]))
            if i==2:
                rows.append({"File Name":str(j),"Laterality":"L","View":"CC","Bi-Rads":["0","4a","4b","6"][j%4],"ACR":str(1+j%4)})
        if i==0:
            rows.append({**rows[0],"abnormality id":"2","abnormality type":"calcification","pathology":"MALIGNANT"})
        file="clinical.xlsx" if i==1 else "annotations.csv"
        if i==1:pd.DataFrame(rows).to_excel(source/file,index=False)
        else:pd.DataFrame(rows).to_csv(source/file,sep=";" if i==2 else ",",index=False)
        configs[name]=dict(root=str(source),version="synthetic-v1",annotations=[dict(path=file,official_split="train")])
    return configs


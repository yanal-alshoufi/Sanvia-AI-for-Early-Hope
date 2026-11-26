"""Read native source annotations. All paths and optional schema bindings come from configuration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pydicom

from sanvia.stage2.stage1_artifacts import local_path, require

SIDES={"L":"L","LEFT":"L","R":"R","RIGHT":"R"}


def table(path: Path, columns: set[str], sep: str=",") -> pd.DataFrame:
    """Validate the release header before interpreting any rows."""
    frame=(pd.read_excel(path,dtype=str,keep_default_na=False) if path.suffix.lower() in {".xls",".xlsx"}
           else pd.read_csv(path,sep=sep,dtype=str,keep_default_na=False,encoding="utf-8-sig"))
    frame.columns=frame.columns.str.strip()
    missing=columns-set(frame)
    require(not missing,f"{path} is missing required columns {sorted(missing)}. Available: {list(frame)}")
    return frame


def header(path: Path) -> dict:
    ds=pydicom.dcmread(path,stop_before_pixels=True)
    return dict(patient=str(ds.get("PatientID","")).strip(),study=str(ds.get("StudyInstanceUID","")).strip(),
                image=str(ds.get("SOPInstanceUID","")).strip(),side=str(ds.get("ImageLaterality",ds.get("Laterality",""))),
                view=str(ds.get("ViewPosition","")),series=str(ds.get("SeriesInstanceUID","")),description=str(ds.get("SeriesDescription","")))


class Adapter:
    def __init__(self, cfg: dict):
        self.cfg=cfg
        self.root=Path(cfg["root"]).resolve()
        self.annotations=cfg["annotations"]
        require(self.root.is_dir() and self.annotations and cfg.get("version"), f"Supply an authorized root, version and annotation files for {self.root}")

    def annotation_path(self, entry: dict) -> Path:
        return Path(self.root / entry["path"])

    def images(self):
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".dcm",".dicom"}:
                yield path,header(path)

    def records(self) -> list[dict]:
        raise NotImplementedError


class CBIS(Adapter):
    """Use the case-description patient's ID and full-image column across mass and calcification files."""
    def records(self):
        records=[]
        series_index=None
        for entry in self.annotations:
            path=self.annotation_path(entry)
            part=entry["official_split"]
            require(part in {"train","test"},"Declare the official CBIS partition of every case description")
            frame=table(path,{"patient_id","left or right breast","image view","abnormality id","abnormality type","pathology","image file path","cropped image file path","ROI mask file path"})
            density=next((k for k in ("breast_density","breast density") if k in frame),None)
            require(density is not None,f"Missing CBIS breast density field in {path}")
            for row in frame.to_dict("records"):
                rel=row["image file path"].strip()
                require(rel and rel not in {row["cropped image file path"].strip(),row["ROI mask file path"].strip()},"Full image and lesion derivative share a path")
                image=local_path(self.root,rel)
                if not image.exists():
                    if series_index is None:
                        series_index={}
                        for candidate,meta in self.images():
                            series_index.setdefault(meta["series"],[]).append(candidate)
                    candidates=series_index.get(Path(rel).parent.name,[])
                    require(len(candidates)==1,f"Missing or ambiguous CBIS full-image series for {rel}")
                    image=candidates[0]
                h=header(image)
                require(h["description"].strip().lower() not in {"cropped images","roi mask images"},f"CBIS derivative in full image column: {rel}")
                side=SIDES.get(row["left or right breast"].strip().upper())
                view=row["image view"].strip()
                if h["side"]:
                    require(side==SIDES.get(h["side"].upper()),"CBIS laterality disagrees with DICOM")
                if h["view"]:
                    require(view==h["view"],"CBIS view disagrees with DICOM")
                records.append(dict(patient=row["patient_id"].strip(),study=h["study"],image=h["image"],side=side,view=view,path=str(image),
                                    lesion=row["abnormality type"]+"-"+row["abnormality id"],pathology=row["pathology"],finding=row["abnormality type"],
                                    density=row[density],official_split=part,annotation=path.name,image_type="full",roi=row["ROI mask file path"],
                                    crop=row["cropped image file path"]))
        return records


class CMMD(Adapter):
    """Attach a unilateral clinical record only to its documented side and unambiguous examination."""
    def records(self):
        require(len(self.annotations)==1,"Configure one CMMD clinical workbook")
        path=self.annotation_path(self.annotations[0])
        frame=table(path,{"ID1","LeftRight","classification"})
        clinical={}
        for row in frame.to_dict("records"):
            key=(row["ID1"].strip(),SIDES.get(row["LeftRight"].strip().upper()))
            require(key[0] and key[1] and key not in clinical,"Ambiguous or repeated CMMD patient/side row")
            clinical[key]=row["classification"]
        records=[]
        matched=set()
        studies={}
        for img,h in self.images():
            side=SIDES.get(h["side"].upper())
            key=(h["patient"],side)
            label=clinical.get(key,"")
            if key in clinical:
                matched.add(key)
                require(studies.setdefault(key,h["study"])==h["study"],"CMMD diagnosis spans multiple examinations without an exam key")
            records.append(dict(**{k:h[k] for k in ("patient","study","image","view")},side=side,path=str(img),pathology=label,
                                image_type="full",annotation=path.name,official_split="unspecified"))
        require(set(clinical)==matched,"CMMD clinical records lack matching DICOM patient and laterality")
        return records


class INbreast(Adapter):
    """Use native file numbers for linkage and DICOM identifiers for grouping."""
    def records(self):
        require(len(self.annotations)==1,"Configure one INbreast annotation file")
        entry=self.annotations[0]
        path=self.annotation_path(entry)
        frame=table(path,{"File Name","Laterality","View","Bi-Rads","ACR"},entry.get("separator",";"))
        index={}
        for img,h in self.images():
            key=img.stem.split("_",1)[0]
            require(key not in index,f"INbreast file number {key} maps to multiple DICOM files")
            index[key]=(img,h)
        result=[]
        for row in frame.to_dict("records"):
            key=row["File Name"].strip()
            require(key in index,f"Missing INbreast full mammogram for file number {key}")
            img,h=index.pop(key)
            side=SIDES.get(row["Laterality"].strip().upper())
            view=row["View"].strip()
            require((not h["side"] or SIDES.get(h["side"].upper())==side) and (not h["view"] or h["view"]==view),"INbreast DICOM/annotation side or view conflict")
            result.append(dict(**{k:h[k] for k in ("patient","study","image")},view=view,side=side,path=str(img),birads=row["Bi-Rads"],
                               density=row["ACR"],image_type="full",annotation=path.name,official_split="unspecified"))
        require(not index,"INbreast DICOM files without matching annotation rows")
        return result


class BCDR(Adapter):
    """Release-bound FM/DM adapter. Require explicit audited semantic and column bindings."""
    def records(self):
        require(self.cfg.get("schema_reference") and self.cfg.get("tasks"),"BCDR requires release-specific schema evidence and task definitions")
        records=[]
        for entry in self.annotations:
            mapping=entry["columns"]
            require({"patient","study","image","side","view","path","image_type"}<=mapping.keys(),"Incomplete BCDR column bindings")
            path=self.annotation_path(entry)
            frame=table(path,set(mapping.values()),entry.get("separator",","))
            for original in frame.to_dict("records"):
                row={k:original[v] for k,v in mapping.items()}
                for key,values in entry.get("vocabulary",{}).items():
                    row[key]=values[row[key]]
                row["path"]=str(local_path(self.root,row["path"]))
                row.setdefault("official_split","unspecified")
                records.append(dict(row,annotation=path.name))
        return records


ADAPTERS={"CBIS-DDSM":CBIS,"CMMD":CMMD,"INbreast":INbreast,"BCDR-FM":BCDR,"BCDR-DM":BCDR}


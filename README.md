# Generative_AI_for_3d_modeling

## Data Transformation

### Command Sequence (.json/.h5) to .step
file: DeepCAD/utils/export2step.py  

Parameters:  
* **src : str, default=None**  
source file or folder (takes every .json/.h5 file in the directory as input)
* **form: str, default=h5, choices=[h5, json]**  
select file format of input (json or h5)  

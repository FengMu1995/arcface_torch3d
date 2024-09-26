# arcface_torch3d
The  repo reproduced the idea "He et al. - Enhancing Face Recognition With Self-Supervised 3D.pdf"

This code is adding a auxiliary face reconstruction to backbone in face feature extraction model based on Arcface so that the depth and albedo features extraction ability can be enhanced.  Moreover, the auxiliary can be trained by self-supervision, no need of extra label-making.

A experiment based on ArcFace in which backbone was chosen as resnet18 model was made.
The experiment aim to evaluate the trained model and the original model(without auxiliary branch) on LFW, CFP_FP, AgeDB, IJB-C and myown face dataset, it is quite an obvious improvement,
So I'd like to commit the code to github.


## Future Attemp
while self-supervising the face reconstruction, label-supervising the auto-encoded face depth-map can reap a better training weight.



## Ackknowledgement
The pro. is based on arcface_torch(https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)
The idea is from "He et al. - Enhancing Face Recognition With Self-Supervised 3D.pdf"

Thank all these contributions.


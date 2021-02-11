FROM  mltooling/ml-workspace-gpu:0.9.1

RUN pip3 install keras matplotlib scikit-learn pandas bokeh pydot librosa scikit-image moviepy tensorflow-graphics-gpu
RUN pip3 install holoviews==1.13.4
RUN apt update && apt-get install -y graphviz
RUN pip3 install opencv-python morecvutils
RUN pip3 install tensorflow-datasets
RUN pip3 install tensorflow_federated
RUN pip3 install nest_asyncio
RUN conda install -c plotly plotly=4.8.1 --yes
RUN pip3 install torch_optimizer
RUN pip3 install ai.cs
RUN pip3 install wandb
RUN pip3 install comet-ml

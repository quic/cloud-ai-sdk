# Introduction 
Docker is a product that allows users to build, test, and deploy applications through software containers. Docker for Cloud AI 100 packages the Platform SDK, Apps SDK (x86 only), libraries, system tools, etc., which enables the user to navigate the inference workflow seamlessly. 

The Docker scripts are in the Apps SDK in the ```tools/docker-build``` folder. The scripts to build a QAic Docker image are composed of the following structure.
```
  ├── build_image.sh
  ├── caffe
  │   ├── detection-output.patch
  │   ├── Dockerfile.ubuntu18
  │   ├── makefile.config.patch
  │   └── mkldnn.patch
  ├── config
  │   ├── aarch64
  │   │   ├── Dockerfile.centos8
  │   │   └── Dockerfile.ubuntu18
  │   ├── qaic_pytools
  │   │   ├── aimet.dockerfile
  │   │   └── pytools.dockerfile
  │   ├── qms_agent
  │   │   └── qms_agent.dockerfile
  │   └── x86_64
  │       ├── Dockerfile.centos8
  │       ├── Dockerfile.triton
  │       └── Dockerfile.ubuntu18
  ├── README.md
```

```Note: Ubuntu18 ocnfig file works for Ubuntu20 as well.```

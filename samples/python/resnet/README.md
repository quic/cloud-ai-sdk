# Classification model
In this example, we run a resnet50 Classification model imported from torchvision library on AIC100.

# Table of contents  
1. [Ovrview](#introduction)  
2. [AIC100](#paragraph1) 
    1. [Compilation](#subparagraph1)  
    2. [Execution](#subparagraph1)  
3. [APIs used](#paragraph2)  

## Overview

Import the model from torchvision (or any other opensource library)


## AIC100

### Compilation

qaic-exec

### Execution

qaicrt api

## APIs Used

#### Check device ID ``qaicrt.Util``

```python
  getAicVersion(self: qaicrt.Util) -> Tuple[qaicrt.QStatus, int, int, str, str]
```  

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `id` | `string` | **Required**. ID of the item you're requesting |

###### Example:

```python
  getAicVersion(self: qaicrt.Util) -> Tuple[qaicrt.QStatus, int, int, str, str]
```  
#### Get item

~~~http
  POST /api/items
~~~

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `name`  | `string` | **Required**. Name of a new item |
| `price` | `number` | **Required**. Name of a new item |  
 
## Appendix  
Any additional information goes here  

 
## Documentation  
[Documentation](https://linktodocumentation)  
 
## FAQ  

#### Question 1  

Answer 1  

#### Question 2  

Answer 2  


# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

def main(bs, pl, cl, cores, socs):
    data = '''{
    "connections": [
        {
            "devices": '''+str(list(range(socs))).replace(" ","")+''',
            "type": "p2p"
        }
    ],
    "partitions": [
        {
            "name": "Partition0",
            "devices": [
                {
                    "deviceId": 0,
                    "numCores": '''+str(cores)+'''
                }'''
    for i in range(1,socs):
        data += ''',
                {
                    "deviceId": '''+str(i)+''',
                    "numCores": '''+str(cores)+'''
                }'''
    data+='''
            ]
        }
    ]
}
    '''

    with open('mdp.json', 'w') as file: file.write(data)


    data = '''{
	"specializations": [
		{
			"batch_size": "'''+str(bs)+'''",
			"seq_len": "'''+str(pl)+'''",
			"ctx_len": "'''+str(cl)+'''"            
		}'''
    if pl>1:
        data +=''',
        {
			"batch_size": "'''+str(bs)+'''",
			"seq_len": "1",
			"ctx_len": "'''+str(cl)+'''"
		}'''
    data +='''
	]
}
        '''

    with open('specializations.json', 'w') as file: file.write(data)


if __name__ == "__main__":
    import argparse

    argp = argparse.ArgumentParser()
    argp.add_argument("--bs", type=int, default=1, help="batchsize")
    argp.add_argument("--pl", type=int, default=64, help="prompt length")
    argp.add_argument("--cl", type=int, default=256, help="context length")
    argp.add_argument("--cores", type=int, default=14, help="cores")
    argp.add_argument("--socs",  type=int, default=4,  help="socs" )

    args = argp.parse_args()
    main(**vars(args))

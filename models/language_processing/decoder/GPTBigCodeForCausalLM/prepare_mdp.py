def main(cores, socs):
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

if __name__ == "__main__":
    import argparse

    argp = argparse.ArgumentParser()
    argp.add_argument("--cores", type=int, default=14, help="cores")
    argp.add_argument("--socs",  type=int, default=4,  help="socs" )

    args = argp.parse_args()
    main(**vars(args))

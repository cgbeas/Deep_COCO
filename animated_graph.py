from apscheduler.schedulers.blocking import BlockingScheduler
import os
import paramiko

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

LOCAL_DIR = os.getcwd()
FILE_PATH = '/home/ec2-user/data/text.txt'
FILE_PATH2 = '/home/ec2-user/data/sample_data.txt'
KEY_PATH = '../../AWS_Free_Tier.pem'
hostname = 'ec2-52-38-78-126.us-west-2.compute.amazonaws.com'
username = 'ec2-user'
key = paramiko.RSAKey.from_private_key_file(KEY_PATH)

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)


### Using Local File ###

# def animate(i):
#     pullData = open("abc.txt","r").read()
#     print(pullData)
#     dataArray = pullData.split('\n')
#     xar = []
#     yar = []
#     for eachLine in dataArray:
#         if len(eachLine)>1:
#             x,y = eachLine.split(',')
#             xar.append(int(x))
#             yar.append(int(y))
#     ax1.clear()
#     ax1.plot(xar,yar)
# ani = animation.FuncAnimation(fig, animate, interval=1000)
# plt.show()

### Using remote file ###

def animate(i):

    # accessing remote host
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh.connect(hostname, username=username, pkey=key)
    sftp = ssh.open_sftp()

    # accesing remote file
    remote_file = sftp.open(FILE_PATH2).read()

    # decoding bytes-like objects
    remote_file = remote_file.decode()

    # pullData = open("abc.txt","r").read()
    dataArray = remote_file.split('\n')
    xar = []
    yar = []

    for eachLine in dataArray:
        if len(eachLine)>1:
            x,y = eachLine.split(',')
            xar.append(int(x))
            yar.append(int(y))
    ax1.clear()
    ax1.plot(xar,yar)

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
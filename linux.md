
## find . -name "*.so"  -delete
. 表示当前目录，-name "*.so" 表示查找后缀为 .so 的文件，-type f 表示只查找文件类型的对象（而不是目录、链接等），-delete 表示删除找到的文件对象。
此命令会递归删除当前目录及其子目录中所有后缀为 .so 的文件，注意谨慎操作，避免误删其他文件。如果只需要查找这些文件，可以去掉 -delete 选项，只保留 find . -name "*.so" -type f 即可。

## nohup 
nohup 是一个 Unix 和类 Unix 系统中的命令，用于在用户终端会话结束时继续运行进程。它通常用于在后台运行长时间的任务，即使用户注销或关闭终端，任务仍然会继续运行。

将输出重定向到文件
bash
复制
nohup sh train.sh > output.log 2>&1 &
> output.log：将标准输出重定向到 output.log 文件。
2>&1：将标准错误重定向到标准输出。
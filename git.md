# clone 
```
git clone ssh://cywang@10.3.2.212:29418/Magik/FxModelZoo
git clone ssh://cywang@10.3.2.212:29418/Magik/MagiKTrainingKit or Inference or TransformKit
git clone http://magik:crMbZ6cyXjzqbm405JDx0vu27GAl8Y+2ZCQ4cciWaw@60.173.195.78:8083/gerrit/magik/magik-toolkit
```
# 提交至ModelZoo
```console
# set user-name and email
git config --global user.name'cywang'
git config --global user.email'ace.cywang@ingenic.com'
git branch -a 
git checkout branch0
git status
git log
git add file0
git checkout file0:not change file0
git cz --> fix([modelzoo][alphapose])
git push origin HEAD:refs/for/refs/heads/FX
git push origin HEAD:refs/for/refs/heads/alphapose
# 如果git cz后打开文件，
commitizen init cz-conventional-changelog --save --save-exact
    if error:
    npm init --yes
    commitizen init cz-conventional-changelog --save --save-exact
git stash
git stash save "describe for your change"
git stash list
git stash apply stash{num}
git stash pop == git stash apply stash{0}
git reset --soft HEAD^
# 查看stash中param_inference.cc较现在的改动
git diff --staged stash@{0} -- quantize/param_inference/param_inference.cc 
# 查看stash所有改动的文件
git stash show stash@{0} 

```

### 使用过程发现
### 历史修改
要查看特定文件 quantize/param_inference/reduction_op_param_inference.cc 的历史提交记录，你可以使用 git log 命令，并指定文件路径作为参数。以下是具体的命令和步骤：

<!-- 查看文件的修改历史： -->
git log -- quantize/param_inference/reduction_op_param_inference.cc
这个命令会显示文件 quantize/param_inference/reduction_op_param_inference.cc 的所有历史提交记录
。

<!-- 查看历史记录的简洁版本： -->
git log --oneline -- quantize/param_inference/reduction_op_param_inference.cc
使用 --oneline 选项可以查看历史记录的简洁版本，每个提交只显示一行
。

<!-- 查看每次提交变化的大小： -->
git log --stat -- quantize/param_inference/reduction_op_param_inference.cc
--stat 选项可以显示每次提交变化的文件列表和文件修改的统计信息
。

<!-- 显示每次提交的差异： -->
git log -p -- quantize/param_inference/reduction_op_param_inference.cc
-p 选项相当于 --full-diff，显示每次提交的差异
。

<!-- 查看某一次提交中的文件变化： -->
git show <commitID> -- quantize/param_inference/reduction_op_param_inference.cc
替换 <commitID> 为你想要查看的具体提交的ID，这个命令可以显示某一次提交中文件的变化
。

这些命令可以帮助你查看文件 quantize/param_inference/reduction_op_param_inference.cc 的历史提交记录和每次提交的具体变化。

## 本次存在未merge的commit
```console
commit d584b4a44970dd0f4e32d86d06a791c60d814465(最新)
Author: yanshaofu <sfyan@example.com>
Date:   Mon Mar 31 11:52:38 2025 +0800

    fix([dev-nna3]): fix fp data format bug
    
    Change-Id: I40aaa609999022a8894e4149705da9f8e7bf3c01
commit 568269c7e2230fac5c2167bece2716dfda28b9ba(本地未合并)
Author: cywang <ace.cywang@ingenic.com>
Date:   Wed Mar 26 12:20:19 2025 +0800
    fix([magiktrainingkit][dev-nna3]): 1.adjust FP16s E0 new

commit 2ea469c7e2230fac5c2167bece2716dfda28b9ba
Author: cywang <ace.cywang@ingenic.com>
Date:   Wed Mar 26 12:20:19 2025 +0800

    fix([magiktrainingkit][dev-nna3]): 1.adjust FP16s E0 2. adjust deconv op
    
    1.adjust FP16s E0 2. adjust deconv op
    
    Change-Id: I837a81700b6931961bce06bde61795f2b5f60560

```
放弃本地更改
如果你想放弃本地更改，可以重置本地分支到远程分支的状态：
git fetch origin
git reset --hard origin/dev-NNA3
git pull

## 提交原则
1.readme示例代码应该直接
2.transample
3.gitignore
## remove the useless things
```
find ./ -type f -name "*.py" -exec sed -i 's/\t/    /g' {} \; -exec sed -i 's/[ \t]*$//g' {} \; -exec sed -i 's/ *$//g' {} \;

sed -i 's/\t/    /g' *.py：这部分命令使用sed工具来将所有 .py 文件中的制表符（\t）替换为四个空格。-i标志表示直接在文件中进行替换操作，而不是输出到标准输出。*.py表示所有的 .py 文件。

sed -i 's/[ \t]*$//g' *.py：这部分命令用于去除所有 .py 文件中每行末尾的空格和制表符。这个命令通过正则表达式s/[ \t]*$//g实现。这里的[ \t]*表示匹配零个或多个空格或制表符，$表示行尾。

sed -i 's/ *$//g' *.py：最后这部分命令用于去除所有 .py 文件中每行末尾的空格。它也使用了类似的正则表达式s/ *$//g，其中*匹配零个或多个空格。


sed -i 's/\t/    /g' *.cpp *.py; sed -i 's/[ \t]*$//g' *.cpp *.py;sed -i 's/ *$//g' *.cpp *.py
sed -i 's/\t/    /g' *.py *.txt; sed -i 's/[ \t]*$//g' *.py *.txt; sed -i 's/ *$//g' *.py *.txt
sed -i 's/\t/    /g' *.md; sed -i 's/[ \t]*$//g' *.md; sed -i 's/ *$//g'  *.md

sed -i 's/\t/    /g' *.cpp *.cu; sed -i 's/[ \t]*$//g' *.cpp *.cu;sed -i 's/ *$//g' *.cpp *.cu
```

## 删除所有要删除的文件
```
git ls-files --deleted -z | xargs -0 git rm
```
这个命令会将git ls-files --deleted获取到的文件列表作为参数传递给 xargs 命令，并使用 git rm 命令进行删除操作。-z 参数和 -0 参数用于处理文件名中可能包含空格或其他特殊字符的情况。
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
git stash list
git stash apply stash{num}
git stash pop == git stash apply stash{0}
git reset --soft HEAD^
```
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
```

## 删除所有要删除的文件
```
git ls-files --deleted -z | xargs -0 git rm
```
这个命令会将git ls-files --deleted获取到的文件列表作为参数传递给 xargs 命令，并使用 git rm 命令进行删除操作。-z 参数和 -0 参数用于处理文件名中可能包含空格或其他特殊字符的情况。
使用softmax回归实现MNIST手写数字识别
使用三层神经网络结构，隐层以及输出层都采用sigmoid激活函数
代价函数形式为：
\begin{split}\begin{split}
 J(\theta) & = \frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} \left[
-y_k^{(i)} \log((h_{\theta}(x^{(i)}))_k)
-(1 - y_k^{(i)}) \log(1 - (h_{\theta}(x^{(i)}))_k)
\right] \\
& + \frac2{\lambda}{m} \left[
\sum_{j=1}^{25} \sum_{k=1}^{400} (\Theta_{j,k}^{(1)})^2 +
\sum_{j=1}^{10} \sum_{k=1}^{25} (\Theta_{j,k}^{(2)})^2 \right]
\end{split}\end{split}

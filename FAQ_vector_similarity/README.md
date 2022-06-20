# Faiss的基本介绍和用法学习

Faiss的全称是Facebook AI Similarity Search。 它是一个开源库，针对高维空间中的海量数据，提供了高效且可靠的检索方法。

基本用法：

    index = faiss.IndexFlatL2(d)/faiss.IndexFlatIP--d代表向量的维度
    index.add()--numpy格式，float32
    index.search()--向量，找k个

faiss.IndexFlatL2(d):

    numpy.sqrt(numpy.sum(numpy.square(searched - query))),平方

faiss.IndexFlatIP(d):

    np.dot(query.T, searched)

基本方法是暴力搜索，即遍历每一个向量进行计算的，如果index里面的向量过于多了，就会有问题

faiss.IndexIVFFlat：构建索引

    quantizer = faiss.IndexFlatL2(d)  # 量化器
    index = faiss.IndexIVFFlat(quantizer, d, nlist--聚类中心个数, faiss.METRIC_L2)
    index.train(vectors) # 要对这堆向量算出聚类中心
    index.add(vectors)
    index.nprobe = 5 # 修改查找的聚类中心，默认的时候是先去找nprobe个聚类中心，然后再比较这里面的所有
    注意nprobe和nlist两个数值要匹配使用，一般是成比例增大

但是768维的向量，总体来说还是比较大的，而在faiss当中有压缩算法，可以对向量进行压缩

    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer,d,nlist--聚类中心个数,m--切割成m份，8)   #注意最后一个参数nbits_per_idx要小于等于8
    index.train(xb)
    index.add(xb)
    index.nprobe = 3 # 搜索的聚类个数

三种index索引情况：

    search_one_query("孩童中耳炎流黄水要如何医治", normal_index, 2)
    句子生成向量时间， 1.0599839687347412
    索引向量时间， 0.0010004043579101562
    Out[4]: (array([[0.     , 7.86909]], dtype=float32), array([[ 28, 755]], dtype=int64))
    search_one_query("孩童中耳炎流黄水要如何医治", center_index, 2)
    句子生成向量时间， 1.0576732158660889
    索引向量时间， 0.0009987354278564453
    Out[5]: (array([[0.     , 7.86909]], dtype=float32), array([[ 28, 755]], dtype=int64))
    search_one_query("孩童中耳炎流黄水要如何医治", compression_index, 2)
    句子生成向量时间， 1.0959687232971191
    索引向量时间， 0.0
    Out[6]: 
    (array([[5.6189117, 9.614075 ]], dtype=float32),
     array([[ 28, 755]], dtype=int64))

整体来说，对于比较大的n，IndexIVFFlat肯定优于IndexFlatL2

其次随着n进一步增大，需要对原始向量进行压缩，从而节省存储空间

下来以后好好理解一下这两个blog：

https://blog.csdn.net/rangfei/article/details/108177652

https://blog.csdn.net/qq_33283652/article/details/116976900

使用单卡GPU创建索引：

    res = faiss.StandardGpuResources()  # use a single GPU
    # build a flat (CPU) index
    index_flat = faiss.IndexFlatL2(d)
    # make it into a gpu index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

使用多张卡创建GPU索引：

    ngpus = faiss.get_num_gpus()
    cpu_index = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
        cpu_index
    )

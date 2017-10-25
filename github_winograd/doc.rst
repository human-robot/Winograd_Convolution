winograd layer 
---------------

WinogradLayer类

::

属性
-------

::

 private:
  int m_group_;                # group 
  int m_batchSize;             # batchSize
  
  int m_bottom_dim_;// par size
  int m_top_dim_;
  
  // The following variables are initialized in WeightAlign

  int tile_h_in_;              # input tile size  H
  int tile_w_in_;              # input tile size  W
  int tile_h_out_;             # output tile size H'
  int tile_w_out_;             # output tile size W'
  int ntiles_h_;               # number of tiles  H有几个tile 
  int ntiles_w_;               # number of tiles  W有几个tile
  
  int conv_in_channels_; //ic  # conv input channel
  int conv_out_channels_;//oc  # conv output channel
  
  int m_iH;                    # winograd input H
  int m_iW;                    # winograd input W
  
  int m_oH;                    # winograd output H
  int m_oW;                    # winograd output W
  
  int m_kH;//kernel            # kernel H
  int m_kW;                    # kernel W
  int m_sH;//stride            # stride H
  int m_sW;                    # stride W
  
  int m_pad;                   # pad 
  bool m_bias;                 # bias 

 private:

  Dtype* m_inputOrg;           # conv input 
  const Dtype* m_weightOrg;    # conv weight 
  
  Dtype* m_winogradWeight; // support NCHW storage ＃　winograd weight
  Dtype* m_winogradInput;     # winograd input 
  
  Dtype* m_col_buff;//buffer  # wingograd column buffer 
  
  WINOGRAD_ALG m_alg;         # 哪种类型的winograd kerne F(6,3) or F(4,3) or F(4,5)

构造函数
----------
对参数的不同配置

::

  WinogradLayer(WINOGRAD_ALG alg, 
              int batch_size, 
  	      int iH, 
  	      int iW, 
  	      int iC, 
  	      int kH, 
  	      int kW, 
  	      int sH, 
  	      int sW, 
  	      int oC, 
  	      int pad, 
  	      bool bias = true) : m_alg(alg) {

  m_iH = iH;
  m_iW = iW;
  conv_in_channels_ = iC;
  m_kH = kH;
  m_kW = kW;
  m_sH = sH;
  m_sW = sW;
  conv_out_channels_ = oC;
  m_pad = pad; // pad_h = pad_w
  m_bias = bias;
  
  m_batchSize = batch_size;
  m_group_ = 1;
  
  m_bottom_dim_ = 0;// default batch =1
  m_top_dim_ = 0;
  
  m_winogradWeight = NULL;
  m_winogradInput = NULL;
  
  // Output width.
  m_oW = (m_iW + m_pad * 2 - m_kW) / m_sW + 1;   // (W + 2*P - KW)/Stride + 1
  m_oH = (m_iH + m_pad * 2 - m_kH) / m_sH + 1;   // (H + 2*P - KH)/Stride + 1

  if (alg == WT_8X8_F_6X6_3X3) {
  	tile_h_in_ = 8;
  	tile_w_in_ = 8; /* input tile size */
  	tile_h_out_ = tile_h_in_ - m_kH + 1;
  	tile_w_out_ = tile_w_in_ - m_kW + 1; /* output tile size */
  	ntiles_h_ = (PUBLIC_TOOL::max(m_iH + m_pad - tile_h_in_ + 1, m_oH) + tile_h_out_ - 1) / tile_h_out_;
  	ntiles_w_ = (PUBLIC_TOOL::max(m_iW + m_pad - tile_w_in_ + 1, m_oW) + tile_w_out_ - 1) / tile_w_out_;
  
  }
  else if (alg == WT_6X6_F_4X4_3X3) {
  
  	tile_h_in_ = 6;
  	tile_w_in_ = 6; /* input tile size */
  	tile_h_out_ = tile_h_in_ - m_kH + 1;
  	tile_w_out_ = tile_w_in_ - m_kW + 1; /* output tile size */
  	ntiles_h_ = (PUBLIC_TOOL::max(m_iH + m_pad - tile_h_in_ + 1, m_oH) + tile_h_out_ - 1) / tile_h_out_;
  	ntiles_w_ = (PUBLIC_TOOL::max(m_iW + m_pad - tile_w_in_ + 1, m_oW) + tile_w_out_ - 1) / tile_w_out_;
  
  }
  else throw("convolution algorithm error!");

  }

method
--------
get_inference_cpu:类似与caffe framework中每层layer的推理函数forward函数

::

  const std::shared_ptr<Dtype> get_inference_cpu(Dtype* data, const Dtype* par, Dtype* col_buff) {
  
  	m_inputOrg = data;
  	m_weightOrg = par;
  	m_col_buff = col_buff;
  
        //resOut 计算结果大小，为之分配空间．oH*oW*out_C
  	std::shared_ptr<Dtype> resOut = std::shared_ptr<Dtype>(new Dtype[m_oH*m_oW*conv_out_channels_]);
  	//trans weight to winograd domain
  	trans_weight2wiongrad();  //Gg
  
  	for (int n = 0; n < m_batchSize; n++) {
  		//trans input to winograd domain
  		trans_input2winograd(m_inputOrg + n*m_bottom_dim_, m_col_buff);//BTd
  		// Convolution in Winograd domain
  		winograd_conv(); //Gg*BTd
  		// Transform back to time domain	
  		trans2spatial(resOut.get() + n*this->m_top_dim_);
  		//bias
  		if (this->m_bias) {
  			int base = conv_in_channels_ * conv_out_channels_ * m_kW * m_kH;
  			const Dtype* bias = &par[base];
  			this->forward_cpu_bias(resOut.get() + n * this->m_top_dim_, bias);
  		}
  	}
  	return  resOut;
  }


trans_weight2wiongrad 
------------------------
::

 void trans_weight2wiongrad() {// weight: hwcn --> cn hw
 
 	// transform weights to Winograd domain
 	if (!m_winogradWeight) 
                m_winogradWeight = 
 		new Dtype[conv_in_channels_*conv_out_channels_* tile_h_in_ *tile_w_in_];
 
 	PUBLIC_TOOL::dlm_cpu_gemm(CblasNoTrans, CblasTrans,
 		tile_h_in_*tile_w_in_, (conv_in_channels_ / m_group_)*conv_out_channels_, m_kH*m_kW,
 		(Dtype)1,
 		Winograd_Kron::getInstance(m_alg, WINOGRAD_G)->get().get(),
 		m_weightOrg,
 		(Dtype)0,
 		m_winogradWeight);			
 
 }


dlm_cpu_gemm
-------------------

C = alpha*op(A)*op(B) + beta*C 

::

   void dlm_cpu_gemm(const CBLAS_TRANSPOSE TransA,
   const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
   const float alpha, const float* A, const float* B, const float beta,
   float* C) 
   {
   int lda = (TransA == CblasNoTrans) ? K : M;
   int ldb = (TransB == CblasNoTrans) ? N : K;
   cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
   			ldb, beta, C, N);
   }

cblas_sgemm
-------------

C = alpha*op(A)*op(B) + beta*C 

::

  void cblas_sgemm(
  const enum CBLAS_ORDER Order, //数据存储是行主序还是列主序
  const enum CBLAS_TRANSPOSE TransA,
  const enum CBLAS_TRANSPOSE TransB, 
  const int M, //A行
  const int N, //B列
  const int K, //A列,B行
  const float alpha, 
  const float *A,
  const int lda, //A列
  const float *B, 
  const int ldb, //B列
  const float beta, 
  float *C, 
  const int ldc  //C列
  );

  const enum CBLAS_ORDER Order，这是指的数据的存储形式，在CBLAS的函数中无论一维还是二维数据都是用一维数组存储，这就要涉及是行主序还是列主序，在C语言中数组是用 行主序，fortran中是列主序。我还是习惯于是用行主序，所以这个参数是用CblasRowMajor，如果是列主序的话就是 CblasColMajor。



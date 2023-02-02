//builtin.module  {
  //builtin.func @main() {
    //%0 = tiny.constant dense<[[72, 101], [108, 32]]> : tensor<2x2xi32>
    //tiny.print %0 : tensor<2x2xi32>
    //tiny.return
  //}
//}
builtin.module  {
  builtin.func @main() {
    %0 = "tiny.constant"() {value = dense<[[72, 101, 43], [108, 32, 17]]> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
    %1 = tiny.transpose(%0 : tensor<2x3xi32>) to tensor<2x3xi32>
    %2 = tiny.transpose(%1 : tensor<2x3xi32>) to tensor<2x3xi32>
    "tiny.print"(%2) : (tensor<2x3xi32>) -> ()
    tiny.return
  }
}


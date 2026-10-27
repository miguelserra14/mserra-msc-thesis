// Queries over the CPG of exemplo.c.
// Run with: joern --script consultas.sc --param cpgFile=out/cpg.bin

@main def exec(cpgFile: String) = {
  importCpg(cpgFile)

  println("\n== 1. Functions defined in the file ==")
  cpg.method.isExternal(false).filterNot(_.name == "<global>").foreach { m =>
    println(s"${m.name}  (lines ${m.lineNumber.getOrElse(-1)}-${m.lineNumberEnd.getOrElse(-1)})")
  }

  println("\n== 2. Calls inside each function, with line numbers ==")
  cpg.call.filterNot(_.name.startsWith("<operator>")).foreach { c =>
    println(s"line ${c.lineNumber.getOrElse(-1)}: ${c.method.name} -> ${c.code}")
  }

  println("\n== 3. Node count per function (what becomes the graph of each sample) ==")
  cpg.method.isExternal(false).filterNot(_.name == "<global>").foreach { m =>
    println(s"${m.name}: ${m.ast.size} AST nodes, ${m.cfgNode.size} CFG nodes")
  }

  println("\n== 4. Who calls whom (call graph, the interprocedural part) ==")
  cpg.method.isExternal(false).filterNot(_.name == "<global>").foreach { m =>
    println(s"${m.name} calls: ${m.callee.name.filterNot(_.startsWith("<operator>")).dedup.l.mkString(", ")}")
  }

  println("\n== 5. Data flow: does argv reach the 2nd argument of strcpy? ==")
  def source = cpg.method.name("main").parameter.name("argv")
  def sink   = cpg.call.name("strcpy").argument(2)
  sink.reachableByFlows(source).p.foreach(println)
}

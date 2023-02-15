val scala3Version = "3.2.2"

val compilerOptions = Seq(
    "-encoding",
    "utf8",
    "-feature",
    "-unchecked",
    "-deprecation",
    "-language:implicitConversions"
    // "-explain",
    // "-Wvalue-discard",
    //"-Vprofile",
)

lazy val `kyo-settings` = Seq(
    scalaVersion := scala3Version,
    fork         := true,
    scalacOptions ++= compilerOptions,
    scalafmtOnCompile := true,
    organization      := "io.getkyo",
    homepage          := Some(url("https://getkyo.io")),
    licenses          := List("Apache-2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0")),
    developers := List(
        Developer(
            "fwbrasil",
            "Flavio Brasil",
            "fwbrasil@gmail.com",
            url("https://github.com/fwbrasil/")
        )
    ),
    ThisBuild / sonatypeCredentialHost := "s01.oss.sonatype.org",
    sonatypeRepository                 := "https://s01.oss.sonatype.org/service/local"
)

lazy val genOpt = TaskKey[Unit]("genOpt", "")

lazy val genOptState: State => State = { s: State =>
  "genOpt" :: s
}

lazy val kyo = (project in file("."))
  .aggregate(
      `kyo-core`,
      `kyo-core-opt1`,
      `kyo-core-opt2`,
      `kyo-core-opt3`,
      `kyo-bench`,
      `kyo-zio`,
      `kyo-direct`
  )
  .settings(
      name := "kyo",
      `kyo-settings`,
      publishArtifact := false,
      genOpt := {
        genOpt(1)
        genOpt(2)
        genOpt(3)
      },
      Global / onLoad := {
        val old = (Global / onLoad).value
        genOptState compose old
      }
  )

val zioVersion = "2.0.6"

lazy val `kyo-core-settings` = `kyo-settings` ++ Seq(
    libraryDependencies += "com.lihaoyi"   %% "sourcecode"        % "0.3.0",
    libraryDependencies += "dev.zio"       %% "izumi-reflect"     % "2.2.2",
    libraryDependencies += "org.slf4j"      % "slf4j-api"         % "2.0.6",
    libraryDependencies += "org.jctools"    % "jctools-core"      % "4.0.1",
    libraryDependencies += "dev.zio"       %% "zio-test"          % zioVersion   % Test,
    libraryDependencies += "dev.zio"       %% "zio-test-magnolia" % zioVersion   % Test,
    libraryDependencies += "dev.zio"       %% "zio-test-sbt"      % zioVersion   % Test,
    libraryDependencies += "dev.zio"       %% "zio-prelude"       % "1.0.0-RC16" % Test,
    libraryDependencies += "dev.zio"       %% "zio-laws-laws"     % "1.0.0-RC16" % Test,
    libraryDependencies += "org.scalatest" %% "scalatest"         % "3.2.15"     % Test
)

def genOpt(i: Int) = {
  val origin = new File("kyo-core/src/")
  val dest   = new File(s"kyo-core-opt$i/src/")
  IO.copyDirectory(origin, dest)

  def inlining(file: File): Unit = {
    if (file.isDirectory) file.listFiles.foreach(inlining)
    else {
      var original = IO.read(file)
      var content  = original
      for (i <- 1 to i)
        content = content.replaceAllLiterally(s"/*inline(${4 - i})*/", "inline")
      IO.write(file, content)
    }
  }
  inlining(dest)
}

lazy val `kyo-core` = project
  .in(file("kyo-core"))
  .settings(
      name := "kyo-core",
      `kyo-core-settings`
  )

lazy val `kyo-core-opt1` = project
  .in(file(s"kyo-core-opt1"))
  .settings(
      name := s"kyo-core-opt1",
      `kyo-core-settings`,
      scalafmtOnCompile := false
  )

lazy val `kyo-core-opt2` = project
  .in(file(s"kyo-core-opt2"))
  .settings(
      name := s"kyo-core-opt2",
      `kyo-core-settings`,
      scalafmtOnCompile := false
  )

lazy val `kyo-core-opt3` = project
  .in(file(s"kyo-core-opt3"))
  .settings(
      name := s"kyo-core-opt3",
      `kyo-core-settings`,
      scalafmtOnCompile := false
  )

lazy val `kyo-direct` = project
  .in(file("kyo-direct"))
  .dependsOn(`kyo-core` % "test->test;compile->compile")
  .settings(
      name := "kyo-direct",
      `kyo-settings`,
      libraryDependencies += "com.github.rssh" %% "dotty-cps-async" % "0.9.16"
  )

lazy val `kyo-zio` = project
  .in(file("kyo-zio"))
  .dependsOn(`kyo-core` % "test->test;compile->compile")
  .settings(
      name := "kyo-zio",
      `kyo-settings`,
      libraryDependencies += "dev.zio" %% "zio" % zioVersion
  )

lazy val `kyo-bench` = project
  .in(file("kyo-bench"))
  .enablePlugins(JmhPlugin)
  .dependsOn(`kyo-core-opt3`)
  .settings(
      name := "kyo-bench",
      `kyo-settings`,
      libraryDependencies += "org.typelevel" %% "cats-effect" % "3.3.12",
      libraryDependencies += "dev.zio"       %% "zio"         % zioVersion
  )

testFrameworks += new TestFramework("zio.test.sbt.ZTestFramework")

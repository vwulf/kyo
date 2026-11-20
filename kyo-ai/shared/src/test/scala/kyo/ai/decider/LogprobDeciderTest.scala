package kyo.ai.decider

import kyo.*
import kyo.Decider.Query
import kyo.Decider.internal.*
import kyo.ai.Config
import kyo.ai.Context
import kyo.ai.Context.*
import kyo.ai.DeciderConfig

object LogprobDeciderTest:
    enum Tool derives Schema, CanEqual:
        case Shell, Database, Manual

    case class Fs(manifest: Chunk[String], lastWrite: String) derives Schema, CanEqual

    /** What one request carried: its Authorization header and its raw body, in arrival order. */
    case class Seen(authorization: Maybe[String], body: String) derives CanEqual

    case class Msg(role: String, content: String) derives Schema
    case class Req(messages: Chunk[Msg]) derives Schema

    /** An OpenAI-compatible `/v1/chat/completions` stand-in. Each request is answered by the first script
      * whose marker occurs in the request's last message (the question), with that script's first-token
      * alternatives; a request no script matches gets a 400 naming the fixture, so a test bug reads as one.
      */
    def withServer[A, S](scripts: Seq[(String, String)])(f: (String, AtomicRef[Chunk[Seen]]) => A < S)(using
        Frame
    ): A < (S & Async & Scope & Abort[HttpBindException | HttpRouteException]) =
        AtomicRef.init(Chunk.empty[Seen]).map { seen =>
            val route =
                HttpRoute.postRaw("v1/chat/completions")
                    .request(_.headerOpt[String]("authorization").bodyText)
                    .response(_.bodyText)
                    .handler { req =>
                        seen.updateAndGet(_.append(Seen(req.fields.authorization, req.fields.body))).andThen {
                            val question = Json.decode[Req](req.fields.body) match
                                case Result.Success(r) => r.messages.lastMaybe.fold("")(_.content)
                                case _                 => ""
                            scripts.find((marker, _) => question.contains(marker)) match
                                case Some((_, reply)) => HttpResponse.ok(reply)
                                case None             =>
                                    HttpResponse(HttpStatus(400)).addField(
                                        "body",
                                        s"""{"error":"no script matches the question: $question"}"""
                                    )
                            end match
                        }
                    }
            HttpServer.initWith(HttpServerConfig.default)(route) { server =>
                f(s"http://127.0.0.1:${server.port}/v1", seen)
            }
        }

    /** A chat completion whose first token is the first alternative, carrying every alternative with its
      * probability as a log-probability, and the given usage.
      */
    def reply(alternatives: Seq[(String, Double)], promptTokens: Int = 100, cachedTokens: Int = 0): String =
        val tops       = alternatives.map((t, p) => s"""{"token":"$t","logprob":${math.log(p)}}""").mkString(",")
        val (first, p) = alternatives.head
        s"""{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"$first"},""" +
            s""""logprobs":{"content":[{"token":"$first","logprob":${math.log(p)},"top_logprobs":[$tops]}]},"finish_reason":"length"}],""" +
            s""""usage":{"prompt_tokens":$promptTokens,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":$cachedTokens}}}"""
    end reply

    def close(a: Double, b: Double): Boolean = math.abs(a - b) < 1e-9

end LogprobDeciderTest

/** The logprob backend end to end against a fake OpenAI-compatible endpoint: what each request carries,
  * how labels read off the first token become answers, how a batch fans out over the shared prefix, and
  * how it fails.
  */
class LogprobDeciderTest extends kyo.test.Test[Any]:
    import LogprobDeciderTest.*

    def config(url: String, decider: DeciderConfig => DeciderConfig = identity): Config =
        Config.Anthropic.default.apiKey("unused").retrySchedule(Schedule.done)
            .decider(decider(DeciderConfig.Logprob.default.apiUrl(url).modelName("served-model")))

    val fs = Fs(Chunk("a.txt", "b.txt"), "b.txt")

    "a noul reads Y and N off the first token, whatever word-start mark the tokenizer adds" in {
        val tokens = Seq("ĠY" -> 0.6, " Y" -> 0.15, "▁N" -> 0.2, "Maybe" -> 0.05)
        withServer(Seq("Is it broken?" -> reply(tokens))) { (url, seen) =>
            LLM.run(config(url))(Decider.noul(fs, "Is it broken?").map(p => Decider.check(fs, "Is it broken?").map(c => (p, c)))).map {
                (probability, checked) =>
                    seen.get.map { requests =>
                        assert(close(probability, 0.75 / 0.95), s"Y and its variants over Y and N: $probability")
                        assert(checked)
                        assert(requests.size == 2)
                        val body = requests(0).body
                        assert(body.contains(""""model":"served-model""""), body)
                        assert(body.contains(""""max_tokens":1""") && body.contains(""""logprobs":true"""), body)
                        assert(body.contains(""""top_logprobs":20"""), body)
                        assert(
                            body.contains(""""role":"user","content":"{\"manifest\":[\"a.txt\",\"b.txt\"],\"lastWrite\":\"b.txt\"}""""),
                            s"the one-shot's context is the shared user message: $body"
                        )
                        assert(body.contains("Question: Is it broken?"), body)
                        assert(body.contains("Answer with exactly one label: Y, N."), body)
                    }
            }
        }
    }

    "a choice labels its options A, B, C in the order given and returns the most probable option" in {
        val tokens = Seq("B" -> 0.7, "A" -> 0.2, "C" -> 0.1)
        withServer(Seq("Which tool?" -> reply(tokens))) { (url, seen) =>
            LLM.run(config(url))(Decider.query(Query.choice("Which tool?", Tool.values.toSeq))).map { decision =>
                seen.get.map { requests =>
                    assert(decision.best == Tool.Database)
                    assert(decision.probabilities.map(_._1) == Chunk(Tool.Shell, Tool.Database, Tool.Manual))
                    assert(decision.probabilities.map(_._2).zip(Seq(0.2, 0.7, 0.1)).forall((a, b) => close(a, b)), decision.toString)
                    assert(requests(0).body.contains("A: Shell\\nB: Database\\nC: Manual"), requests(0).body)
                }
            }
        }
    }

    "a score labels its levels 0, 1, 2 lowest first and returns the probability-weighted position" in {
        val tokens = Seq("1" -> 0.5, "2" -> 0.5)
        withServer(Seq("How healthy?" -> reply(tokens))) { (url, seen) =>
            LLM.run(config(url))(Decider.score("How healthy?", Seq("Healthy", "Degraded", "Corrupt"))).map { position =>
                seen.get.map { requests =>
                    assert(close(position, 1.5), s"0 * 0 + 1 * 0.5 + 2 * 0.5: $position")
                    assert(requests(0).body.contains("ordered from lowest to highest"), requests(0).body)
                    assert(requests(0).body.contains("0: Healthy\\n1: Degraded\\n2: Corrupt"), requests(0).body)
                }
            }
        }
    }

    "a batch sends one request per question over the same prefix, the first alone and before the rest" in {
        val lines   = Seq("first line", "second line", "third line")
        val scripts = lines.zip(Seq(0.9, 0.5, 0.1)).map((line, p) => line -> reply(Seq("Y" -> p, "N" -> (1.0 - p))))
        withServer(scripts) { (url, seen) =>
            LLM.run(config(url))(Decider.batch(fs, lines.map(line => Query.noul(s"Is '$line' relevant?")))).map { answers =>
                seen.get.map { requests =>
                    assert(answers.zip(Seq(0.9, 0.5, 0.1)).forall((a, b) => close(a, b)), answers.toString)
                    assert(requests.size == 3)
                    assert(requests(0).body.contains("first line"), "the first question is sent alone, before the others")
                    val prefixes = requests.map(r => r.body.take(r.body.indexOf("Judge the conversation above")))
                    assert(prefixes.forall(_ == prefixes(0)), s"every request shares the prefix: $prefixes")
                    assert(prefixes(0).contains("\\\"manifest\\\""), prefixes(0))
                }
            }
        }
    }

    "a batch's usage sums its requests, cached prompt tokens included" in {
        val scripts = Seq(
            "one?"   -> reply(Seq("Y" -> 1.0), promptTokens = 100, cachedTokens = 0),
            "two?"   -> reply(Seq("N" -> 1.0), promptTokens = 100, cachedTokens = 90),
            "three?" -> reply(Seq("Y" -> 1.0), promptTokens = 100, cachedTokens = 90)
        )
        withServer(scripts) { (url, _) =>
            val decider   = DeciderConfig.Logprob.default.apiUrl(url).modelName("served-model")
            val cfg       = config(url)
            val questions = Chunk("one?", "two?", "three?").map(q => Question.Noul(Structure.Value.Str(q), Absent, Absent, Absent))
            LLM.run(cfg)(AI.init.map(_ => LogprobDecider.decide(cfg, decider, Context.empty, questions))).map { reply =>
                assert(reply.answers.map {
                    case Answer.Noul(p) => p
                    case other          => -1.0
                } == Chunk(1.0, 0.0, 1.0))
                assert(reply.usage == AIStats(300L, Present(180L), 3L, Absent, 3))
            }
        }
    }

    "the conversation is the shared prefix: system, user and assistant turns, tool use rendered as text" in {
        val context = Context(Chunk(
            SystemMessage("You manage files."),
            UserMessage("Delete a.txt", Absent),
            AssistantMessage("", Chunk(Call(CallId("c1"), "rm", """{"path":"a.txt"}"""))),
            ToolMessage(CallId("c1"), "removed")
        ))
        assert(LogprobDecider.conversation(context) == Chunk(
            LogprobDecider.ChatMessage("system", "You manage files."),
            LogprobDecider.ChatMessage("user", "Delete a.txt"),
            LogprobDecider.ChatMessage("assistant", "\n[called rm with {\"path\":\"a.txt\"}]"),
            LogprobDecider.ChatMessage("user", "[result of call c1]\nremoved")
        ))
    }

    "the Authorization header rides only when a key is set" in {
        withServer(Seq("Keyed?" -> reply(Seq("Y" -> 1.0)), "Open?" -> reply(Seq("Y" -> 1.0)))) { (url, seen) =>
            for
                _        <- LLM.run(config(url, _.apiKey("local-key")))(Decider.check("Keyed?"))
                _        <- LLM.run(config(url))(Decider.check("Open?"))
                requests <- seen.get
            yield
                assert(requests(0).authorization == Present("Bearer local-key"))
                assert(requests(1).authorization == Absent)
            end for
        }
    }

    "a reply whose first token is none of the labels fails as AIDecodeException, naming the token" in {
        withServer(Seq("Is it?" -> reply(Seq("<think>" -> 0.9, "The" -> 0.1)))) { (url, _) =>
            Abort.run[AIException](LLM.run(config(url))(Decider.check("Is it?"))).map { result =>
                assert(
                    result.failure.exists(e =>
                        e.isInstanceOf[AIDecodeException] && e.getMessage.contains("none of the labels Y, N") &&
                            e.getMessage.contains("<think>")
                    ),
                    result.toString
                )
            }
        }
    }

    "a choice with more options than labels fails before any request" in {
        withServer(Seq.empty) { (url, seen) =>
            val options = (1 to LogprobDecider.maxLabels + 1).map(i => s"option $i")
            for
                result   <- Abort.run[AIException](LLM.run(config(url))(Decider.choose("Which?", options)))
                requests <- seen.get
            yield
                assert(
                    result.failure.exists(e =>
                        e.isInstanceOf[AIInvalidQuestionException] && e.getMessage.contains(s"labels at most ${LogprobDecider.maxLabels}")
                    ),
                    result.toString
                )
                assert(requests.isEmpty)
            end for
        }
    }

    "a choice uses lowercase labels past Z, up to 52 options" in {
        val options = (1 to LogprobDecider.maxLabels).map(i => s"option $i")
        withServer(Seq("Which?" -> reply(Seq("z" -> 1.0)))) { (url, seen) =>
            LLM.run(config(url))(Decider.choose("Which?", options)).map { chosen =>
                seen.get.map { requests =>
                    assert(chosen == "option 52")
                    assert(requests(0).body.contains("a: option 27"), requests(0).body)
                }
            }
        }
    }

end LogprobDeciderTest

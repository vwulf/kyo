package kyo.ai.decider

import kyo.*
import kyo.Decider.internal.*
import kyo.ai.Config
import kyo.ai.Context
import kyo.ai.Context.*
import kyo.ai.DeciderConfig

/** The decider backend that reads each answer from one token's log-probabilities on an OpenAI-compatible
  * chat completions endpoint (`POST /chat/completions` with `logprobs`), one request per question, all
  * sharing the conversation as a common prefix.
  *
  * Every request carries the same messages (the conversation, rendered as role/content turns) followed by
  * one final user message that asks a single question and lists its labels: `Y`/`N` for a noul, `A`, `B`,
  * ... for a choice's options (then `a`, `b`, ..., at most 52), `0`, `1`, ... for a score's levels, lowest
  * first. The request asks for one output token at temperature 0 with the top 20 log-probabilities, and
  * the answer is the distribution of those tokens over the labels: each label's probability is the sum
  * over the returned tokens that read as it (surrounding whitespace and the tokenizer's word-start marks
  * ignored), normalized over the labels. A label outside the returned top 20 counts as 0; when no label
  * appears at all the answer fails as `AIDecodeException`. The distribution becomes an answer exactly as
  * the completion backend's does (best key, weighted level, entropy-based confidence).
  *
  * Why one request per question: each answer is an independent judgment over the same state, and the
  * requests differ only after the shared prefix. On a server that caches prefixes (vLLM's automatic prefix
  * caching, SGLang's radix cache, OpenAI's prompt caching) the conversation is prefilled once and every
  * further question pays only for its own suffix and one decoded token. The first question is sent alone
  * so its prefill populates the cache; the rest then go out concurrently, bounded by the decider's meter
  * (else the config's), and the server can batch their single decode steps. The reported usage sums the
  * requests and carries the server's cached prompt tokens when it reports them, which is how a caller
  * sees the sharing work. The price of independence is that no answer conditions on another; a question
  * whose answer depends on an earlier one belongs in a later decision.
  *
  * The model must answer in its first token: a reasoning model that opens with a thinking block spends its
  * one token there and its labels never appear, so serve a non-reasoning model or one with thinking
  * disabled. Images are dropped (a label read off a token cannot use them), tool calls and results are
  * rendered as text, and the key is optional: a local server usually needs none, and the `Authorization`
  * header is sent only when one is set or the provider's variable supplies it. Transport is
  * [[DeciderTransport]], under the decider's own timeout, meter and schedule when set.
  */
private[kyo] object LogprobDecider extends Decider.Backend:

    /** How many alternatives the request asks for per token: the most OpenAI and vLLM's default serve. */
    private[kyo] val topLogprobs = 20

    /** The most options a choice can take here: one single-character label per option. */
    private[kyo] val maxLabels = 52

    private val letters: Chunk[String] = Chunk.from(('A' to 'Z') ++ ('a' to 'z')).map(_.toString)

    def decide(config: Config, decider: DeciderConfig, context: Context, questions: Chunk[Question])(using
        Frame
    ): Reply < (LLM & Async & Abort[AIGenException]) =
        unlabeled(decider, questions) match
            case Present(detail) => Abort.fail(AIInvalidQuestionException(detail))
            case Absent          =>
                Log.debug(
                    s"kyo-ai decide backend=${decider.provider.name} model=${decider.modelName} " +
                        s"questions=${questions.size} messages=${context.messages.size}"
                ).andThen(apiKey(decider)).map { key =>
                    val headers = Seq("content-type" -> "application/json") ++ key.fold(Seq.empty[(String, String)])(k => Seq("Authorization" -> s"Bearer $k"))
                    fanOut(config, decider, headers, conversation(context), questions)
                }
    end decide

    // A choice with more options than there are labels, as the failure's detail; checked before any request.
    private def unlabeled(decider: DeciderConfig, questions: Chunk[Question]): Maybe[String] =
        Maybe.fromOption(questions.zipWithIndex.collectFirst {
            case (Question.Choice(_, options), i) if options.size > maxLabels =>
                s"question ${i + 1} offers ${options.size} options; the ${decider.provider.name} decider labels at most $maxLabels"
        })

    // The first request alone warms the server's prefix cache; the rest then share it, concurrently.
    private def fanOut(
        config: Config,
        decider: DeciderConfig,
        headers: Seq[(String, String)],
        shared: Chunk[ChatMessage],
        questions: Chunk[Question]
    )(using Frame): Reply < (LLM & Async & Abort[AIGenException]) =
        def ask(question: Question): (Answer, AIStats) < (LLM & Async & Abort[AIGenException]) =
            val body = Json.encode(Request(decider.modelName, shared.append(prompt(question)), top_logprobs = topLogprobs))
            DeciderTransport.post(config, decider, s"${decider.apiUrl.stripSuffix("/")}/chat/completions", headers, body, Absent)
                .map(response => Abort.get(decode(response, question)))
        end ask
        questions.headMaybe match
            case Absent        => Reply(Chunk.empty, AIStats.empty)
            case Present(head) =>
                ask(head).map { first =>
                    Async.foreach[AIGenException, Question, (Answer, AIStats), LLM](questions.drop(1), questions.size)(ask).map { rest =>
                        val all = Chunk(first).concat(rest)
                        Reply(all.map(_._1), all.foldLeft(AIStats.empty)((acc, a) => acc.add(a._2)))
                    }
                }
        end match
    end fanOut

    // The key when one is set or the provider's variable supplies it; Absent otherwise, which a local
    // server accepts.
    private def apiKey(decider: DeciderConfig)(using Frame): Maybe[String] < (Async & Abort[AIGenException]) =
        decider.apiKey match
            case Present(key) => Present(key)
            case Absent       => Config.read(decider.provider.keyName)

    /** The labels a question offers, in key order, each with the text that says what it stands for: a
      * noul's criteria (or a plain yes/no reading), a choice's option (its key, then its description when it
      * has one), a score's level.
      */
    private[kyo] def labels(question: Question): Chunk[(String, String)] =
        question match
            case Question.Noul(_, whenTrue, whenFalse, _) =>
                Chunk(
                    "Y" -> whenTrue.fold("yes, it holds")(text),
                    "N" -> whenFalse.fold("no, it does not hold")(text)
                )
            case Question.Choice(_, options) =>
                letters.zip(options.zipWithIndex.map { case ((key, description), i) =>
                    description match
                        case Structure.Value.Null   => key
                        case Structure.Value.Str(s) => s"$key ($s)"
                        case other                  => if key == s"c$i" then text(other) else s"$key ${text(other)}"
                })
            case Question.Score(_, levels) => levels.zipWithIndex.map((level, i) => i.toString -> text(level))
        end match
    end labels

    // The keys answerOf expects, in label order: true/false, the option keys, the level indices.
    private def keys(question: Question): Chunk[String] =
        question match
            case _: Question.Noul              => Chunk("true", "false")
            case Question.Choice(_, options)   => options.map(_._1)
            case Question.Score(_, levels)     => levels.zipWithIndex.map((_, i) => i.toString)

    private def text(value: Structure.Value): String =
        value match
            case Structure.Value.Str(s) => s
            case other                  => Json.encode(other)

    private[kyo] def prompt(question: Question): ChatMessage =
        val instructions = question match
            case Question.Noul(i, _, _, _) => i
            case Question.Choice(i, _)     => i
            case Question.Score(i, _)      => i
        val offered = labels(question)
        val ordered = question match
            case _: Question.Score => " The levels are ordered from lowest to highest."
            case _                 => ""
        ChatMessage(
            "user",
            "Judge the conversation above and answer the question below with one label from the list, and nothing else." +
                ordered + "\n\nQuestion: " + text(instructions) + "\n\n" +
                offered.map((label, meaning) => s"$label: $meaning").mkString("\n") +
                "\n\nAnswer with exactly one label: " + offered.map(_._1).mkString(", ") + "."
        )
    end prompt

    /** The conversation as role/content turns: the prefix every question of a decision shares. */
    private[kyo] def conversation(context: Context): Chunk[ChatMessage] =
        context.messages.map {
            case SystemMessage(content)          => ChatMessage("system", content)
            case UserMessage(content, _)         => ChatMessage("user", content)
            case AssistantMessage(content, calls) =>
                val called = calls.map(call => s"\n[called ${call.function} with ${call.arguments}]").mkString
                ChatMessage("assistant", content + called)
            case ToolMessage(callId, content) => ChatMessage("user", s"[result of call ${callId.id}]\n$content")
        }

    // The label a returned token reads as: surrounding whitespace and the word-start marks of byte-level
    // BPE (Ġ) and SentencePiece (▁) tokenizers ignored.
    private def reading(token: String): String =
        token.trim.stripPrefix("Ġ").stripPrefix("▁").trim

    private[kyo] def decode(body: String, question: Question)(using Frame): Result[AIGenException, (Answer, AIStats)] =
        Json.decode[Response](body) match
            case Result.Success(response) =>
                val first = response.choices.headMaybe.flatMap(_.logprobs).flatMap(_.content).flatMap(_.headMaybe)
                first match
                    case Absent =>
                        Result.fail(AIDecodeException("the decider response carries no token log-probabilities"))
                    case Present(token) =>
                        val chosen       = Chunk(TopLogprob(token.token, token.logprob))
                        val alternatives = token.top_logprobs.fold(chosen)(tops => if tops.isEmpty then chosen else tops)
                        val offered      = labels(question).map(_._1)
                        val raw = offered.map(label =>
                            alternatives.filter(t => reading(t.token) == label).map(t => math.exp(t.logprob)).sum
                        )
                        val sum = raw.sum
                        if sum <= 0.0 || sum.isNaN then
                            Result.fail(AIDecodeException(
                                s"none of the labels ${offered.mkString(", ")} is among the first token's top ${alternatives.size} " +
                                    s"(the model began with '${token.token}')"
                            ))
                        else
                            val usage = response.usage.fold(AIStats(0L, Absent, 0L, Absent, 1)) { u =>
                                AIStats(
                                    u.prompt_tokens.getOrElse(0L),
                                    u.prompt_tokens_details.flatMap(_.cached_tokens),
                                    u.completion_tokens.getOrElse(0).toLong,
                                    Absent,
                                    1
                                )
                            }
                            Result.succeed((LLMDecider.answerOf(question, keys(question), raw.map(_ / sum)), usage))
                        end if
                end match
            case Result.Failure(e) => Result.fail(AIDecodeException(s"undecodable decider response: ${e.getMessage}"))
            case Result.Panic(e)   => Result.fail(AIDecodeException(s"undecodable decider response: ${e.getMessage}"))

    // Wire DTOs, field names verbatim. Unknown response fields are ignored on decode.
    private[kyo] case class ChatMessage(role: String, content: String) derives Schema
    private[kyo] case class Request(
        model: String,
        messages: Chunk[ChatMessage],
        max_tokens: Int = 1,
        temperature: Double = 0.0,
        logprobs: Boolean = true,
        top_logprobs: Int
    ) derives Schema
    private[kyo] case class TopLogprob(token: String, logprob: Double) derives Schema
    private[kyo] case class TokenLogprob(token: String, logprob: Double, top_logprobs: Maybe[Chunk[TopLogprob]] = Absent) derives Schema
    private[kyo] case class Logprobs(content: Maybe[Chunk[TokenLogprob]] = Absent) derives Schema
    private[kyo] case class ResponseChoice(logprobs: Maybe[Logprobs] = Absent) derives Schema
    private[kyo] case class PromptTokensDetails(cached_tokens: Maybe[Long] = Absent) derives Schema
    private[kyo] case class Usage(
        prompt_tokens: Maybe[Long] = Absent,
        completion_tokens: Maybe[Int] = Absent,
        prompt_tokens_details: Maybe[PromptTokensDetails] = Absent
    ) derives Schema
    private[kyo] case class Response(choices: Chunk[ResponseChoice], usage: Maybe[Usage] = Absent) derives Schema

end LogprobDecider

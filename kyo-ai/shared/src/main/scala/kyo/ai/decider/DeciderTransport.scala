package kyo.ai.decider

import kyo.*
import kyo.ai.Config
import kyo.ai.DeciderConfig
import kyo.ai.completion.Completion

/** One decision request under the generation loop's transport discipline, shared by the HTTP decider
  * backends.
  *
  * The decider's timeout (else the config's) is a deadline that covers the call and its retries. A
  * non-2xx response is classified into the module's leaves before the retry clause sees it, with the
  * endpoint's request id (when the backend names a header for it) ahead of the body, so a long rejection
  * keeps it. A rate limit's `Retry-After` is waited out under the deadline before the schedule's own
  * backoff, transient leaves retry on the decider's schedule (else the config's), and the decider's meter
  * (else the config's) bounds concurrency. A closed meter is a panic, as it is for a generation.
  *
  * @see
  *   [[TypeSafeDecider]] and [[LogprobDecider]], its callers
  */
private[kyo] object DeciderTransport:

    def post(
        config: Config,
        decider: DeciderConfig,
        url: String,
        headers: Seq[(String, String)],
        body: String,
        requestIdHeader: Maybe[String]
    )(using Frame): String < (LLM & Async & Abort[AIGenException]) =
        val provider      = decider.provider.name
        val timeout       = decider.timeout.getOrElse(config.timeout)
        val meter         = decider.meter.getOrElse(config.meter)
        val retrySchedule = decider.retrySchedule.getOrElse(config.retrySchedule)
        Clock.now.map(_ + timeout).map { deadline =>
            Async.timeoutWithError[AIGenException, Result[AIGenException, String], LLM](
                timeout,
                Result.Failure(AICompletionTimeoutException(provider, timeout))
            ) {
                Abort.run[AIGenException] {
                    HttpClient.withConfig(_.timeout(timeout)) {
                        Abort.run[Closed] {
                            HttpClient.postTextResponse(url, body, headers, failOnError = false).map { response =>
                                if response.status.isSuccess then (response.fields.body: String < (Sync & Abort[AIGenException]))
                                else
                                    val requestId =
                                        requestIdHeader.flatMap(h => response.headers.get(h)).fold("")(id => s"[request id $id] ")
                                    Completion.statusFailure(provider, "POST", url, response, requestId).map(Abort.fail(_))
                            }.handle(
                                Abort.recover[HttpException](e => Abort.fail(Completion.classifyHttp(provider, e)))(_),
                                meter.run,
                                Completion.retryWithin(deadline, retrySchedule)(_)
                            )
                        }.map {
                            case Result.Success(r) => r
                            case Result.Failure(_) => Abort.panic(AIMeterClosedException())
                            case Result.Panic(ex)  => Abort.panic(ex)
                        }
                    }
                }
            }.map(Abort.get(_))
        }
    end post

end DeciderTransport

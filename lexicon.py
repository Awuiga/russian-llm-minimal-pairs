"""Expandable lexical resources for Russian minimal-pair generation.

The generator uses deterministic templates rather than an LLM. Base lexical
dataclasses document the grammatical features available for future expansion,
while ``RULE_CHOICES`` contains curated neutral examples for the current rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class NounForms:
    """A noun with basic grammatical features and selected case forms."""

    lemma: str
    gender: str
    number: str
    nominative: str
    genitive: str
    dative: str
    accusative: str
    instrumental: str
    prepositional: str


@dataclass(frozen=True)
class AdjectiveForms:
    """Common adjective forms used for agreement-focused extensions."""

    lemma: str
    masculine: str
    feminine: str
    neuter: str
    plural: str


@dataclass(frozen=True)
class VerbForms:
    """A compact verb paradigm for future template expansion."""

    lemma: str
    infinitive: str
    present_singular: str
    present_plural: str
    past_masculine: str
    past_feminine: str
    past_plural: str


@dataclass(frozen=True)
class RuleChoice:
    """A frozen set of placeholder values for one template family."""

    fields: tuple[tuple[str, str], ...]
    comment: str = ""
    difficulty: str = ""

    def as_dict(self) -> dict[str, str]:
        """Return placeholders as a mutable dict for ``str.format``."""

        return dict(self.fields)


def choice(*, comment: str = "", difficulty: str = "", **fields: str) -> RuleChoice:
    """Create a rule choice while keeping declarations compact and readable."""

    return RuleChoice(fields=tuple(fields.items()), comment=comment, difficulty=difficulty)


NOUNS: tuple[NounForms, ...] = (
    NounForms("книга", "feminine", "singular", "книга", "книги", "книге", "книгу", "книгой", "книге"),
    NounForms("шкаф", "masculine", "singular", "шкаф", "шкафа", "шкафу", "шкаф", "шкафом", "шкафе"),
    NounForms("окно", "neuter", "singular", "окно", "окна", "окну", "окно", "окном", "окне"),
    NounForms("статьи", "feminine", "plural", "статьи", "статей", "статьям", "статьи", "статьями", "статьях"),
)

ADJECTIVES: tuple[AdjectiveForms, ...] = (
    AdjectiveForms("новый", "новый", "новая", "новое", "новые"),
    AdjectiveForms("чистый", "чистый", "чистая", "чистое", "чистые"),
    AdjectiveForms("интересный", "интересный", "интересная", "интересное", "интересные"),
    AdjectiveForms("важный", "важный", "важная", "важное", "важные"),
)

VERBS: tuple[VerbForms, ...] = (
    VerbForms("читать", "читать", "читает", "читают", "читал", "читала", "читали"),
    VerbForms("обсудить", "обсудить", "обсудит", "обсудят", "обсудил", "обсудила", "обсудили"),
    VerbForms("писать", "писать", "пишет", "пишут", "писал", "писала", "писали"),
    VerbForms("открыть", "открыть", "откроет", "откроют", "открыл", "открыла", "открыли"),
)

ADVERBIALS: tuple[str, ...] = (
    "внимательно",
    "спокойно",
    "подробно",
    "быстро",
    "аккуратно",
    "заранее",
)

INTRODUCTORY_WORDS: tuple[str, ...] = (
    "Конечно",
    "Вероятно",
    "К счастью",
    "По-моему",
    "Во-первых",
    "Значит",
)


RULE_CHOICES: Mapping[str, tuple[RuleChoice, ...]] = MappingProxyType(
    {
        "subject_verb_agreement": (
            choice(
                subject="опытные врачи",
                correct="обсудили",
                incorrect="обсудил",
                obj="сложный случай",
                adverb="внимательно",
                place="в кабинете",
                time="сегодня",
            ),
            choice(
                subject="молодая учительница",
                correct="объяснила",
                incorrect="объяснили",
                obj="новую тему",
                adverb="подробно",
                place="в классе",
                time="утром",
            ),
            choice(
                subject="один из соседских детей",
                correct="нашел",
                incorrect="нашли",
                obj="старую карту",
                adverb="случайно",
                place="на чердаке",
                time="вчера",
            ),
            choice(
                subject="дежурный инженер",
                correct="проверил",
                incorrect="проверили",
                obj="главный прибор",
                adverb="аккуратно",
                place="в лаборатории",
                time="вечером",
            ),
            choice(
                subject="новые студенты",
                correct="подготовили",
                incorrect="подготовил",
                obj="короткий доклад",
                adverb="заранее",
                place="в аудитории",
                time="сегодня",
            ),
            choice(
                subject="письмо от старшей сестры",
                correct="пришло",
                incorrect="пришла",
                obj="на почту",
                adverb="наконец",
                place="в понедельник",
                time="утром",
            ),
            choice(
                subject="внимательные читатели",
                correct="заметили",
                incorrect="заметил",
                obj="важную ошибку",
                adverb="сразу",
                place="в тексте",
                time="вчера",
            ),
            choice(
                subject="ответ главного редактора",
                correct="появился",
                incorrect="появились",
                obj="на сайте",
                adverb="позже",
                place="после совещания",
                time="днем",
            ),
            choice(
                subject="отчет двух экспертов",
                correct="лежал",
                incorrect="лежали",
                obj="на столе",
                adverb="спокойно",
                place="в приемной",
                time="утром",
            ),
            choice(
                subject="каждая из новых заявок",
                correct="попала",
                incorrect="попали",
                obj="в общий список",
                adverb="сразу",
                place="после проверки",
                time="вечером",
            ),
        ),
        "adjective_noun_agreement": (
            choice(correct="новая", incorrect="новую", noun="книга", verb="лежала", place="на верхней полке", time="весь вечер", time_start="Сегодня"),
            choice(correct="высокий", incorrect="высоком", noun="шкаф", verb="стоял", place="у дальней стены", time="весь день", time_start="Вчера"),
            choice(correct="короткое", incorrect="коротком", noun="сообщение", verb="появилось", place="на школьном сайте", time="вечером", time_start="Утром"),
            choice(correct="интересные", incorrect="интересных", noun="статьи", verb="лежали", place="на рабочем столе", time="до вечера", time_start="Сегодня"),
            choice(correct="важная", incorrect="важной", noun="встреча", verb="прошла", place="в малом зале", time="после обеда", time_start="Вчера"),
            choice(correct="старый", incorrect="старом", noun="мост", verb="стоял", place="над тихой рекой", time="много лет", time_start="Раньше"),
            choice(correct="рабочие", incorrect="рабочих", noun="заметки", verb="остались", place="в общей папке", time="до утра", time_start="Сегодня"),
            choice(correct="свежие", incorrect="свежих", noun="газеты", verb="лежали", place="возле входной двери", time="с утра", time_start="Утром"),
            choice(correct="последние", incorrect="последних", noun="новости", verb="появились", place="на главной странице", time="к вечеру", time_start="Сегодня"),
            choice(correct="письменное", incorrect="письменном", noun="объяснение", verb="осталось", place="в личном деле", time="до проверки", time_start="Вчера"),
        ),
        "case_government": (
            choice(subject="молодая мама", verb="гордилась", correct="сыном", incorrect="сына", adverb="искренне", place="на семейном празднике", time="весь вечер", time_start="Вчера"),
            choice(subject="новый ученик", verb="радовался", correct="успеху", incorrect="успехом", adverb="открыто", place="после трудного задания", time="весь день", time_start="Сегодня"),
            choice(subject="пожилая соседка", verb="интересовалась", correct="историей", incorrect="историю", adverb="давно", place="в районной библиотеке", time="каждую неделю", time_start="Раньше"),
            choice(subject="старший брат", verb="помогал", correct="сестре", incorrect="сестру", adverb="охотно", place="после школьных занятий", time="каждый вечер", time_start="Вчера"),
            choice(subject="опытный водитель", verb="управлял", correct="машиной", incorrect="машину", adverb="спокойно", place="на узкой дороге", time="весь путь", time_start="Сегодня"),
            choice(subject="молодой специалист", verb="доверял", correct="коллеге", incorrect="коллегу", adverb="полностью", place="во время проекта", time="с первого дня", time_start="Сначала"),
            choice(subject="дежурный врач", verb="следил", correct="за порядком", incorrect="за порядок", adverb="внимательно", place="в приемном отделении", time="всю смену", time_start="Ночью"),
            choice(subject="новая сотрудница", verb="сомневалась", correct="в ответе", incorrect="в ответ", adverb="сначала", place="на общем собрании", time="несколько минут", time_start="Утром"),
            choice(subject="опытная юристка", verb="возражала", correct="против решения", incorrect="против решение", adverb="спокойно", place="на общем собрании", time="несколько минут", time_start="Сегодня"),
            choice(subject="молодой врач", verb="надеялся", correct="на помощь", incorrect="на помощью", adverb="тихо", place="после трудной смены", time="весь вечер", time_start="Вчера"),
        ),
        "verb_form": (
            choice(subject="маленькие дети", helper="будут", correct="читать", incorrect="читают", obj="новый рассказ", adverb="вслух", place="в классе", time="завтра утром", time_start="Завтра"),
            choice(subject="старшая сестра", helper="хочет", correct="прочитать", incorrect="прочитает", obj="важное письмо", adverb="спокойно", place="на кухне", time="сегодня вечером", time_start="Сегодня"),
            choice(subject="молодая журналистка", helper="начала", correct="писать", incorrect="писала", obj="короткий отчет", adverb="заранее", place="в кабинете", time="после обеда", time_start="Вчера"),
            choice(subject="новые студенты", helper="решили", correct="обсудить", incorrect="обсудили", obj="общий план", adverb="подробно", place="в аудитории", time="после лекции", time_start="Сегодня"),
            choice(subject="дежурный техник", helper="может", correct="открыть", incorrect="откроет", obj="запасной вход", adverb="быстро", place="в здании", time="ранним утром", time_start="Утром"),
            choice(subject="опытные учителя", helper="стали", correct="повторять", incorrect="повторяют", obj="трудное правило", adverb="медленно", place="на уроке", time="перед тестом", time_start="Сегодня"),
            choice(subject="главный редактор", helper="попросил", correct="исправить", incorrect="исправил", obj="короткую заметку", adverb="сразу", place="в редакции", time="до обеда", time_start="Вчера"),
            choice(subject="молодая врач", helper="должна", correct="проверить", incorrect="проверит", obj="новый анализ", adverb="внимательно", place="в лаборатории", time="после смены", time_start="Сегодня"),
            choice(subject="дежурная медсестра", helper="должна", correct="заполнить", incorrect="заполнит", obj="новую форму", adverb="аккуратно", place="в приемной", time="до обеда", time_start="Сегодня"),
            choice(subject="старшие школьники", helper="смогли", correct="решить", incorrect="решили", obj="трудную задачу", adverb="самостоятельно", place="на занятии", time="к вечеру", time_start="Вчера"),
        ),
        "tsya_tsa": (
            choice(actor="мальчик", correct="учится", incorrect="учиться", context="в новой школе", adverb="старательно", place="в классе", time="с сентября", time_start="Сегодня"),
            choice(actor="юный спортсмен", correct="готовится", incorrect="готовиться", context="к важному старту", adverb="упорно", place="на стадионе", time="каждое утро", time_start="Утром"),
            choice(actor="младшая сестра", correct="смеется", incorrect="смееться", context="над доброй шуткой", adverb="тихо", place="на кухне", time="после ужина", time_start="Вечером"),
            choice(actor="новая команда", correct="собирается", incorrect="собираться", context="в большом зале", adverb="быстро", place="у сцены", time="после обеда", time_start="Сегодня"),
            choice(actor="студент хочет", correct="учиться", incorrect="учится", context="на вечерних курсах", adverb="серьезно", place="в институте", time="в этом году", time_start="Теперь"),
            choice(actor="девочка старается", correct="заниматься", incorrect="занимается", context="музыкой каждый день", adverb="регулярно", place="дома", time="после школы", time_start="Сегодня"),
            choice(actor="помощник должен", correct="вернуться", incorrect="вернется", context="к рабочему столу", adverb="быстро", place="в офисе", time="после звонка", time_start="Сейчас"),
            choice(actor="группа решила", correct="встретиться", incorrect="встретится", context="у главного входа", adverb="заранее", place="возле школы", time="перед поездкой", time_start="Завтра"),
            choice(actor="докладчик собирается", correct="вернуться", incorrect="вернется", context="к главной теме", adverb="скоро", place="на семинаре", time="после вопроса", time_start="Сегодня"),
            choice(actor="новый порядок", correct="утверждается", incorrect="утверждаться", context="приказом директора", adverb="официально", place="в организации", time="с понедельника", time_start="Сейчас"),
        ),
        "ne_ni": (
            choice(prefix="Он", prefix_lc="он", correct="не раз", incorrect="ни раз", suffix="перечитывал этот договор перед встречей", adverb="внимательно", time_start="Вчера"),
            choice(prefix="", prefix_lc="", correct="ни один", incorrect="не один", suffix="участник не опоздал к началу встречи", adverb="обычно", time_start="На встрече"),
            choice(prefix="Брат", prefix_lc="брат", correct="не сразу", incorrect="ни сразу", suffix="нашел старый билет в кармане", adverb="сначала", time_start="Утром"),
            choice(prefix="Секретарь", prefix_lc="секретарь", correct="не", incorrect="ни", suffix="отправил важное письмо до обеда", adverb="еще", time_start="Сегодня"),
            choice(prefix="Студент", prefix_lc="студент", correct="не до конца", incorrect="ни до конца", suffix="понял сложное правило на лекции", adverb="сразу", time_start="Вчера"),
            choice(prefix="Никто", prefix_lc="никто", correct="не", incorrect="ни", suffix="увидел новый анализ в журнале", adverb="пока", time_start="Утром"),
            choice(prefix="Редактор", prefix_lc="редактор", correct="не", incorrect="ни", suffix="заметил лишнюю строку в тексте", adverb="сразу", time_start="Сегодня"),
            choice(prefix="Инженер", prefix_lc="инженер", correct="не", incorrect="ни", suffix="закрыл запасной вход после смены", adverb="случайно", time_start="Вчера"),
            choice(prefix="Мы", prefix_lc="мы", correct="не вполне", incorrect="ни вполне", suffix="согласились с итоговым решением комиссии", adverb="сразу", time_start="Вчера"),
            choice(prefix="", prefix_lc="", correct="ни один", incorrect="не один", suffix="документ не потерялся после переезда", adverb="пока", time_start="После переезда"),
        ),
        "n_nn": (
            choice(correct="стеклянная", incorrect="стекляная", noun="ваза", predicate="стояла", place="на узкой полке", place_tail="у окна", time="весь день", time_start="Сегодня"),
            choice(correct="деревянная", incorrect="деревяная", noun="ложка", predicate="лежала", place="на кухонном столе", place_tail="у плиты", time="после обеда", time_start="Вчера"),
            choice(correct="оловянный", incorrect="оловяный", noun="солдатик", predicate="стоял", place="на книжной полке", place_tail="у лампы", time="много лет", time_start="Раньше"),
            choice(correct="ветреный", incorrect="ветренный", noun="день", predicate="начался", place="в нашем городе", place_tail="у реки", time="рано утром", time_start="Сегодня"),
            choice(correct="румяный", incorrect="румянный", noun="пирог", predicate="остыл", place="на чистой тарелке", place_tail="у окна", time="к вечеру", time_start="Вчера"),
            choice(correct="серебряная", incorrect="серебрянная", noun="ложка", predicate="лежала", place="на белой салфетке", place_tail="у чашки", time="весь вечер", time_start="Сегодня"),
            choice(correct="жареная", incorrect="жаренная", noun="картошка", predicate="осталась", place="на большой сковороде", place_tail="у плиты", time="после ужина", time_start="Вечером"),
            choice(correct="кожаный", incorrect="кожанный", noun="ремень", predicate="лежал", place="в верхнем ящике", place_tail="у двери", time="всю неделю", time_start="Утром"),
            choice(correct="глиняная", incorrect="глинянная", noun="чашка", predicate="стояла", place="на нижней полке", place_tail="у окна", time="с утра", time_start="Сегодня"),
            choice(correct="масляная", incorrect="маслянная", noun="краска", predicate="лежала", place="в закрытой коробке", place_tail="у двери", time="до ремонта", time_start="Вчера"),
        ),
        "fused_separate_spelling": (
            choice(prefix="Ученики работали", prefix_lc="ученики работали", correct="в течение", incorrect="втечение", suffix="долгого урока очень спокойно", adverb="внимательно", time_start="Сегодня"),
            choice(prefix="Дети гуляли", prefix_lc="дети гуляли", correct="несмотря на", incorrect="не смотря на", suffix="холодный ветер во дворе", adverb="радостно", time_start="Вчера"),
            choice(prefix="Брат", prefix_lc="брат", correct="также", incorrect="так же", suffix="помог соседям вечером после работы", adverb="охотно", time_start="Сегодня"),
            choice(prefix="Занятие отменили", prefix_lc="занятие отменили", correct="вследствие", incorrect="в следствие", suffix="сильного дождя перед началом урока", adverb="полностью", time_start="Утром"),
            choice(prefix="Девочка пошла", prefix_lc="девочка пошла", correct="навстречу", incorrect="на встречу", suffix="матери после последнего урока", adverb="быстро", time_start="Сегодня"),
            choice(prefix="Ученики", prefix_lc="ученики", correct="сначала", incorrect="с начала", suffix="прочитали короткий текст вслух на уроке", adverb="внимательно", time_start="Утром"),
            choice(prefix="Группа", prefix_lc="группа", correct="вовремя", incorrect="во время", suffix="закончила общую работу перед обедом", adverb="спокойно", time_start="Вчера"),
            choice(prefix="Мастер сделал", prefix_lc="мастер сделал", correct="вручную", incorrect="в ручную", suffix="новую деталь для старого шкафа", adverb="аккуратно", time_start="Сегодня"),
            choice(prefix="Сосед говорил", prefix_lc="сосед говорил", correct="насчет", incorrect="на счет", suffix="ремонта старой лестницы у подъезда", adverb="спокойно", time_start="Сегодня"),
            choice(prefix="Команда действовала", prefix_lc="команда действовала", correct="втайне", incorrect="в тайне", suffix="от остальных участников проекта", adverb="осторожно", time_start="Вчера"),
        ),
        "complex_sentence_comma": (
            choice(main="Я закрыл окно", main_lc="я закрыл окно", connector="потому что", subordinate="стало холодно", time_tail="к вечеру", time_start="Сегодня"),
            choice(main="Мы вернулись домой", main_lc="мы вернулись домой", connector="когда", subordinate="начался сильный дождь", time_tail="после прогулки", time_start="Вчера"),
            choice(main="Сестра улыбнулась", main_lc="сестра улыбнулась", connector="потому что", subordinate="получила доброе письмо", time_tail="утром", time_start="Сегодня"),
            choice(main="Молодой учитель остановился", main_lc="молодой учитель остановился", connector="когда", subordinate="прозвенел громкий звонок", time_tail="в коридоре", time_start="Вчера"),
            choice(main="Дети обрадовались", main_lc="дети обрадовались", connector="потому что", subordinate="начались долгие каникулы", time_tail="после четверти", time_start="Сегодня"),
            choice(main="Редактор перечитал текст", main_lc="редактор перечитал текст", connector="когда", subordinate="нашел важную ошибку", time_tail="перед печатью", time_start="Утром"),
            choice(main="Мама позвала сына", main_lc="мама позвала сына", connector="когда", subordinate="ужин был готов", time_tail="на кухне", time_start="Вечером"),
            choice(main="Студенты замолчали", main_lc="студенты замолчали", connector="потому что", subordinate="началась новая лекция", time_tail="в аудитории", time_start="Сегодня"),
            choice(main="Я оставил записку", main_lc="я оставил записку", connector="чтобы", subordinate="сосед не волновался", time_tail="вечером", time_start="Сегодня"),
            choice(main="Мы остановились", main_lc="мы остановились", connector="так как", subordinate="дорога стала скользкой", time_tail="после снегопада", time_start="Вчера"),
        ),
        "introductory_comma": (
            choice(intro="Конечно", body="ученики быстро решили простую задачу самостоятельно", time_tail="на уроке", place_tail="в классе"),
            choice(intro="Вероятно", body="сосед скоро вернется домой к ужину", time_tail="после работы", place_tail="к вечеру"),
            choice(intro="К счастью", body="поезд прибыл на станцию вовремя", time_tail="ранним утром", place_tail="без задержки"),
            choice(intro="По-моему", body="доклад получился достаточно ясным для слушателей", time_tail="после обсуждения", place_tail="на семинаре"),
            choice(intro="Во-первых", body="команда проверила все исходные данные заранее", time_tail="перед запуском", place_tail="в лаборатории"),
            choice(intro="Значит", body="мы можем начать обсуждение нового плана", time_tail="после обеда", place_tail="в кабинете", difficulty="subtle"),
            choice(intro="Правда", body="редактор исправил последнюю ошибку в тексте", time_tail="перед печатью", place_tail="для печати", difficulty="subtle"),
            choice(intro="Возможно", body="встреча пройдет в малом зале по плану", time_tail="после обеда", place_tail="сегодня"),
            choice(intro="Впрочем", body="вопрос остался открытым после короткого обсуждения", time_tail="для комиссии", place_tail="на совещании", difficulty="subtle"),
            choice(intro="К примеру", body="эта формулировка подходит для короткого ответа", time_tail="в анкете", place_tail="в задании"),
        ),
        "participial_phrase_comma": (
            choice(main="На столе лежало письмо", main_lc="на столе лежало письмо", participle="подписанное новым директором", time_tail="утром", time_start="Сегодня"),
            choice(main="У окна стоял стул", main_lc="у окна стоял стул", participle="сломанный младшим братом", time_tail="вчера", time_start="Вечером"),
            choice(main="В шкафу висело пальто", main_lc="в шкафу висело пальто", participle="купленное прошлой зимой", time_tail="долго", time_start="Утром"),
            choice(main="На полке лежала тетрадь", main_lc="на полке лежала тетрадь", participle="заполненная аккуратным почерком", time_tail="весь день", time_start="Сегодня"),
            choice(main="В саду стояла скамейка", main_lc="в саду стояла скамейка", participle="покрашенная серой краской", time_tail="после ремонта", time_start="Вчера"),
            choice(main="В комнате висела картина", main_lc="в комнате висела картина", participle="написанная местным художником", time_tail="много лет", time_start="Раньше"),
            choice(main="На столе стояла чашка", main_lc="на столе стояла чашка", participle="наполненная горячим чаем", time_tail="после ужина", time_start="Вечером"),
            choice(main="В папке лежал договор", main_lc="в папке лежал договор", participle="подготовленный юристом заранее", time_tail="до встречи", time_start="Сегодня"),
            choice(main="В конверте лежала справка", main_lc="в конверте лежала справка", participle="выданная районной поликлиникой", time_tail="утром", time_start="Сегодня"),
            choice(main="На стене висел график", main_lc="на стене висел график", participle="составленный старшим инженером", time_tail="до проверки", time_start="Вчера"),
        ),
        "adverbial_participle_comma": (
            choice(gerund="Закончив работу", gerund_lc="закончив работу", main="студент выключил компьютер в кабинете", time_tail="вечером", time_start="Вчера"),
            choice(gerund="Прочитав письмо", gerund_lc="прочитав письмо", main="сестра радостно улыбнулась за столом", time_tail="утром", time_start="Сегодня"),
            choice(gerund="Открыв окно", gerund_lc="открыв окно", main="учитель проветрил класс перед уроком", time_tail="заранее", time_start="Утром"),
            choice(gerund="Проверив список", gerund_lc="проверив список", main="секретарь быстро позвонил всем участникам", time_tail="до обеда", time_start="Сегодня"),
            choice(gerund="Услышав звонок", gerund_lc="услышав звонок", main="дети быстро вошли в класс", time_tail="после перемены", time_start="Вчера"),
            choice(gerund="Получив ответ", gerund_lc="получив ответ", main="редактор продолжил работу над текстом", time_tail="вечером", time_start="Сегодня"),
            choice(gerund="Закрыв дверь", gerund_lc="закрыв дверь", main="сосед поднялся на второй этаж", time_tail="молча", time_start="Утром"),
            choice(gerund="Подготовив отчет", gerund_lc="подготовив отчет", main="инженер заранее отправил файл руководителю", time_tail="после проверки", time_start="Вчера"),
            choice(gerund="Сравнив ответы", gerund_lc="сравнив ответы", main="учитель отметил две похожие ошибки в тетрадях", time_tail="в журнале", time_start="Сегодня"),
            choice(gerund="Заметив опечатку", gerund_lc="заметив опечатку", main="редактор быстро вернул статью автору", time_tail="перед обедом", time_start="Утром"),
        ),
    }
)


def get_choices(rule_id: str) -> tuple[RuleChoice, ...]:
    """Return lexical choices for a rule id."""

    return RULE_CHOICES[rule_id]

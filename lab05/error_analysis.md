# Error analysis

Model: GRU, max_length=100, ngưỡng positive=0.5. Ba mẫu đúng và ba mẫu sai từ lần đánh giá test, đánh số 0–5. Dưới đây giữ nguyên văn review từ dataset, gồm cả nội dung sau giới hạn 100 token; model chỉ đọc 100 token đầu.

Đây là gợi ý kiểm tra do AI hỗ trợ sau khi đọc review và đối chiếu tokenizer/vocabulary. Cần tự làm các phép thử trước khi kết luận nguyên nhân (PDF mục 27).

## Mẫu 0 — Negative → Negative

P(positive)=0.3172; 107 tokens, giữ 100.

**Review gốc:**

> So boring you'll fall asleep after the 20 first minutes. Sorry Mr Boutonnat, I do admire your work (all these beautiful "films" you directed such as "Tristana", "Sans logique" etc...) but here, the plot is extremely... vain ! Except the magnificent photography, everything appears dumb and there's no envy to know what will happen at these "medium" actors. Moreover, the dialogs are minimalists. The famous question "where are the children" is repeated so often it looks like a farce. Believe me, it's a pure waste of time (concerning the plot), and 3 hours is a long long time. Certainly the real reason of this box-office total mess !

review đã có nhiều lời chê rõ, dù xen lời khen đạo diễn và hình ảnh. Bảy token bị cắt chỉ bổ sung ý “box office total mess”, nên phần đánh giá chính vẫn còn.

**Nên kiểm tra:** so sánh xác suất khi bỏ đoạn khen “I do admire…” và “magnificent photography”. Nếu prediction vẫn negative, kết quả phù hợp với việc lời chê chiếm ưu thế; chưa chứng minh model hiểu cấu trúc “but”.

## Mẫu 1 — Negative → Negative

P(positive)=0.2953; 219 tokens, giữ 100.

**Review gốc:**

> end of the world looks like a good movie on the box cover but be warned its bad.being a big fan of mega star actor Christopher lee i was fooled. like tentacles(77)a good cast does not mean it'll be a good film.the beginning looks promising then it turns to tedium,many parts of the movie are too dark to see whats going on.the rest of the cast is shown briefly like;Lew ayres,dean Jagger,and MacDonald Carey as a security guard.i would expect better from Charles band who made good fun movies like the re-animater and dolls.i heard Christopher lee was tricked into making this film.he was told his co stars were going to be Jose ferrer,john carridine,and dean Jagger.well dean Jagger is there but the other two never made it(lucky them)if you make a movie about the end of the world use a bigger budget and better storyline.dean Jagger went on to do two good movies afterwards;alligator(80)and game of death(79) sue Lyon(Lolita)is also in the movie minus her lollipop.if you are a glutton for punishment then watch end of the world i dare you to say awake during the whole movie.1 out of 10.

Cả ba đoạn đều nằm trong review. “1 out of 10” ở cuối bị cắt, nhưng model vẫn dự đoán đúng.

**Nên kiểm tra:** đổi “its bad” thành “its good”, rồi kiểm tra riêng câu “a good cast does not mean it'll be a good film”. Cách này giúp xem model có nhạy với phủ định/tương phản hay chỉ dựa vào vài từ sentiment. Đổi từng đoạn một để dễ đối chiếu.

## Mẫu 2 — Positive → Positive

P(positive)=0.6358; 499 tokens, giữ 100.

**Review gốc:**

> After the success of "Muppet Babies" Warner Brothers chalked up "Tiny Toons". But instead of making Bugs Bunny, Daffy Duck and all the rest of the Looney Toon gang kids, they created new animal characters who were kids with their own distinct personalities but personalities that nonetheless mirrored their predecessors. The leads included Buster & Babs Bunny (no relation, which became their running gag or catch phrase), Plucky Duck, Hampton Pig, Dizzy Devil, Shirley 'the' Loon, Elmira, Montana Max, Furball, Sweety, the rats and assorted animals of Perfecto Prep, and the original Looney Toons cast themselves. The "Tiny Toons" lived in Acme Acres and attended Acme Looniversity, where the Looney Toon gang worked as teachers who served as mentors to the younger generation the ins and out of comedy.<br /><br />During the show's run, various pot shots were taken at the Bush SR. administration, pop culture, and coupled with various other gags and spoofs.<br /><br />Buster & Babs, arguably the show's main characters, as mentioned above, were similar to Bugs Bunny in some respects, but they also had their own differing personality ticks and comic styles, namely, Babs' tendency to impersonate anyone and everyone, while Buster, capable of being a great goof himself, usually played straight man (or straight rabbit) to Babs' antics. Plucky Duck was a virtual copy of Daffy Duck (not screwball Daffy but egomaniac Daffy), with nearly as big an ego as Daffy and just as much of an obsession with upstaging the Buster & Babs as Daffy had with upstaging Bugs, though he usually fell flat on his face in his attempts, yet he remained strangely endearing through out. Hampton was an even more shy version of Porky Pig, and he had the thankless job of playing Porky to Plucky's Daffy. Shirley, the blond duck gal, was a new age valley girl type whom Plucky would go in and out of phases of mocking or vying for her affections. Dizzy was the purple version of the Tasmanian Devil. Furball was the silent Sylvester and Sweety was the pink Tweetie bird. There was also the purple female skunk who longed for a boyfriend and the pint sized versions of Wile Coyote and the Road Runner. Evil was defined in the form of Montana Max, a rich kid who was always out to make a buck or make people's lives miserable. There was also Elmira, a deranged animal lover whom everyone feared. And then there was Godo Dodo, an odd thing-a-ma-gig creature who had no clearly designated species except that he was from "Wacky Land" or something like that.<br /><br />Pop culture references included Batman (quite frequently actually), Michael Jackson, Vanilla Ice, Dances With Wolves, Indiana Jones, Star Trek, Supergirl, fast food joints, the Ten Commandments, the Twilight Zones, Saturday Night Live and even the Simpsons, among others.<br /><br />Not only funny, but it also managed to be warm and touching, something it's successor "Animaniacs" never quite attained. Also followed by "Taz-Mania".

Lời khen rõ nhất ở cuối không được đưa vào model. Review chủ yếu kể tên và mô tả nhân vật; 10/100 token là UNK, gồm “babs”, “hampton”, “elmira”… Dự đoán đúng ở đây chưa đủ cho thấy model hiểu toàn bộ review.

**Nên kiểm tra:** thay “success” bằng một từ trung tính và so với đầu vào là 100 token cuối. Nếu xác suất đổi nhiều, cần xem độ nhạy với từ mở đầu hoặc vị trí lấy context. Đây là phép thử giả thuyết, chưa phải kết quả đã đo.

## Mẫu 3 — Positive → Negative

P(positive)=0.2433; 355 tokens, giữ 100.

**Review gốc:**

> Once a year in America, Saturday morning would give way to new cartoons, usually sometime in America. Just about the time school started.<br /><br />I guess this was to make not having school on Saturdays that more bearable.<br /><br />One Friday night before the premiere of the new Saturday morning cartoons, a movie came on.<br /><br />"At The Earth's Core" I had not a clue who Cushing nor McClure were. Clearly they were not from the same country, so I couldn't grasp why they were traveling together.<br /><br />I do recall seeing the mechanism in the fire-breathing monster's mouth.<br /><br />What connects this movie for me is the opening with the molten metal and then suddenly we get the band playing as we see the factory.<br /><br />How do you end a movie like this? The music again, and two bewlidered cops as the drill appears behind them.<br /><br />Whatever reason, I remember watching this movie more that Friday night than I do any new cartoon that came on the following morning.<br /><br />That was definitely the late seventies.<br /><br />It would be during Jurassic Park mania, cicra '93, I believe, before I would see this movie again.<br /><br />Yes, people in costumes. There was the fire-breather again.<br /><br />Still loved the ending with the two cops.<br /><br />Finally, it would be nearly another decade before I would send off for the movie on DVD.<br /><br />Now I would just watch the beginning and intro.<br /><br />Oh, I do love much of the movie. Of all these sci fi Jules Verne, HG Wells, ERB movies, this one stands as my fave.<br /><br />What I note now is that one of the cops begins turning around a wee bit too soon, then turns back, then he and the other cop are synchronized as they turn inward once again to look behind themselves at the drill.<br /><br />Perhaps for me, this movie stands as the best of the Saturday morning shows, which is why I enjoy it so much.

Review kể hồi ức và sự khó hiểu khi xem phim; các câu thể hiện sự yêu thích đều nằm sau token 100. review chỉ có một UNK (“saturdays”), nên thiếu vocabulary chưa phải giả thuyết nổi bật.

**Nên kiểm tra:** dùng 100 token cuối với cùng checkpoint rồi so xác suất. Nếu chuyển positive, điều đó ủng hộ ảnh hưởng của việc chọn Review. Muốn đánh giá lợi ích của max_length lớn hơn, cần train lại và so trên cùng split. Chưa có cơ sở quy lỗi cho vanishing gradient vì model chưa đọc các lời khen này.

## Mẫu 4 — Negative → Positive

P(positive)=0.7050; 110 tokens, giữ 100.

**Review gốc:**

> Everything you do in this world should make at least a little bit of sense. Unfortunately very little of "Delusion" makes any sense. Jennifer Rubin is adequate in her role as the main squeeze of hit man Kyle Secor. Secor on the other hand overacts to the point of annoyance. Jim Metzler, the embezzling yuppie is very unbelievable as the novice, revenge seeking, adversary. When Jennifer Rubin gives back the money she has carefully been concealing, all credibility flies out the window, and the guns pointing final showdown between Secor and Metzler is beyond ridiculous. Avoid "Delusion", unless you are delusional enough to believe the misguided positive reviews here. - MERK

Các lời chê này đều còn trong Review. Vì vậy, truncation không đủ giải thích lỗi. Chín token trong Review là UNK, phần lớn là tên riêng lặp lại như “rubin”, “secor”, “metzler”. Từ “positive” trong “misguided positive reviews” nằm ở phần bị cắt, nên không thể dùng nó để giải thích prediction.

**Nên kiểm tra:** chạy riêng các câu chê phía trên; sau đó thay tên riêng bằng “the actor” và so xác suất. Nếu câu ngắn đúng nhưng Review sai, cần kiểm tra ảnh hưởng của context/tên riêng. Nếu vẫn sai, xem thêm khả năng học tín hiệu sentiment của model sau 3 epochs. Chưa đủ bằng chứng kết luận lỗi do sequence modeling.

## Mẫu 5 — Positive → Negative

P(positive)=0.4934; 259 tokens, giữ 100.

**Review gốc:**

> "Jaded" should not be considered as en erotic thriller because the sex scenes are not for viewing pleasure unlike movies like "Indecent Behavior" or "Friend Of The Family". <br /><br />"Jaded" is not an easy watch as it's story is difficult to conceive and it's not that common in society. From the beginning of the movie, with the first scene, you expect more strong scenes. <br /><br />The raping scene wasn't (in my opinion) a well created scene. I think that the director didn't use proper camera techniques overall in the movie. That's why it looks sometimes dull and becomes boring almost near the end.<br /><br />The performances are extremely good from all the five female characters. These women gave strong, intense, but believable performances that demonstrate the actresses' dramatic abilities. <br /><br />The legal concepts helped a lot to understand the plot's difficulty because the legal term of sodomy is different from a simple raping. From the point of view of Meg's defense, sodomy needs to be penalized as a raping (a huge crime grade B) although it's a different crime. Pat and Alex's defense claims that Meg enjoyed and asked for it. This is where the movie turns interesting to see who is trying to tell the truth. The only thing for sure is that the sex scene fits into a legal definition of a crime. <br /><br />Watch "Jaded" if you like legal issues that aren't very common in your society. If not, give it a try because it's a very good and strong drama.

Review thiên về phê bình, còn lời khen diễn xuất và kết luận tích cực nằm sau giới hạn. Đây là review có cảm xúc lẫn lộn, với phần chuyển sang khen bị mất. Xác suất chỉ thấp hơn ngưỡng 0.5 khoảng 0.0066 nên prediction cũng sát ranh giới.

**Nên kiểm tra:** so Review với 100 token cuối và với câu kết riêng. Giữ nguyên ngưỡng khi so; không chỉnh ngưỡng dựa trên một lỗi test. Review chỉ có một UNK (“conceive”), nên nên ưu tiên kiểm tra truncation và phân bố sentiment trong review.

## Giới hạn

Các phép thử trên chưa được chạy. Thay đoạn văn hoặc dùng suffix là kiểm tra độ nhạy, không phải bằng chứng chắc chắn về cơ chế bên trong model. Sáu mẫu này cũng không đại diện cho toàn bộ test set.

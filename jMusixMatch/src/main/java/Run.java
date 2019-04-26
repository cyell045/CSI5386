import org.jmusixmatch.MusixMatch;;
import org.jmusixmatch.entity.track.Track;
import org.jmusixmatch.entity.track.TrackData;
import org.jmusixmatch.entity.lyrics.Lyrics;

import java.util.List;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.io.FileWriter;
import java.io.PrintWriter;

/**
 * Created by celineyelle on 2019-03-18.
 */
public class Run {

    public static void main(String[] args){
        String apiKey = "b27b3c0e54527cf0d4cd1b6c4293734c";
        MusixMatch musixMatch = new MusixMatch(apiKey);

        try {
            String fileName = "/Users/celineyelle/IntelliJProjects/jMusixMatch/src/main/resources/output.txt";
            FileWriter fileWriter = new FileWriter(fileName);
            PrintWriter printWriter = new PrintWriter(fileWriter);
            printWriter.print("Track ID \t Topic \t Lyrics \n");


            LinkedHashMap<String, String> topics = new LinkedHashMap<String, String>();
            topics.put("Christmas", "Christmas");
            topics.put("Merry Christmas", "Christmas");
            topics.put("Santa Claus", "Christmas");
            topics.put("Happy Holidays", "Christmas");
            topics.put("Snow", "Christmas");
            topics.put("Dance", "Dance");
            topics.put("Dancefloor", "Dance");
            topics.put("Dancing", "Dance");
            topics.put("Dancin'", "Dance");
            topics.put("Boogie", "Dance");
            topics.put("Bounce", "Dance");
            topics.put("Love", "Love");
            topics.put("Together", "Love");
            topics.put("Family", "Love");
            topics.put("Heart", "Love");
            topics.put("Cupid", "Love");
            topics.put("Kiss", "Love");
            topics.put("Nature", "Nature");
            topics.put("Tree", "Nature");
            topics.put("Forest", "Nature");
            topics.put("Mountain", "Nature");
            topics.put("Land", "Nature");
            topics.put("Sea", "Nature");
            topics.put("Sky", "Nature");


            HashSet<String> titleSet = new HashSet<String>();
            int results = 0;
            String currentTopic = "Christmas";

            for (String topic : topics.keySet()) {
                System.out.println(currentTopic + " - " + topic);
                if (!topics.get(topic).equals(currentTopic)){
                    currentTopic = topics.get(topic);
                    results = 0;
                }

                try {
                    for (int page = 1; page <= 15 & results < 2000; page++) {
                        List<Track> tracks = musixMatch.searchForTracks("", "", "", topic, page, 100, "en", true);

                        for (Track track : tracks) {
                            TrackData trackData = track.getTrack();
                            String trackNameStr = trackData.getTrackName();
                            trackNameStr = trackNameStr.replaceAll("\\(.*\\)", "");

                            if (!titleSet.contains(trackNameStr)) {
                                Lyrics lyrics = musixMatch.getLyrics(trackData.getTrackId());
                                String lyricsStr = lyrics.getLyricsBody();

                                if (!lyricsStr.isEmpty()) {
                                    lyricsStr = lyricsStr.replaceFirst("\\*\\*\\*\\*\\*\\*\\* This Lyrics is NOT for Commercial use \\*\\*\\*\\*\\*\\*\\*", "");
                                    lyricsStr = lyricsStr.replaceAll("\t", " ");
                                    lyricsStr = lyricsStr.replaceAll("\n", " ");
                                    lyricsStr = lyricsStr.replaceAll("\\(1409618290863\\)", "");
                                    lyricsStr = lyricsStr.replaceAll("\\.\\.\\.", "");
                                    printWriter.print(trackData.getCommontrackId() + "\t" + currentTopic  + "\t" + lyricsStr + "\n");
                                    titleSet.add(trackNameStr);
                                    results++;
                                }
                            }
                        }
                    }
                } catch(Exception e){

                }
            }

            printWriter.close();

        }
        catch(Exception e){

        }
    }

}


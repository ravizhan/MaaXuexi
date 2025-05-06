package main

import (
	"crypto/md5"
	"encoding/json"
	"fmt"
	"github.com/hashicorp/go-version"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

func CheckUpdate() (string, error) {
	resp, err := http.Get("https://update.ravi.cool/version")
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP error: %s", resp.Status)
	}
	var data map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return "", err
	}
	v, _ := data["version"].(string)
	return v, nil
}

func GenerateMetadata(dictionary string) (map[string]interface{}, error) {
	metadata := make(map[string]interface{})

	err := filepath.Walk(dictionary, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			relativePath := strings.TrimPrefix(path, dictionary)
			if relativePath == "" {
				return nil
			}
			if strings.HasPrefix(relativePath, "config") || strings.HasPrefix(relativePath, "debug") {
				return nil
			}
			parts := strings.Split(relativePath, string(filepath.Separator))
			current := metadata
			for _, part := range parts[:len(parts)-1] {
				if part == "" {
					continue
				}
				if current[part] == nil {
					current[part] = make(map[string]interface{})
				}
				current = current[part].(map[string]interface{})
			}
			file, err := os.Open(path)
			if err != nil {
				return err
			}
			defer file.Close()
			hash := md5.New()
			_, err = io.Copy(hash, file)
			if err != nil {
				return err
			}
			fileHash := fmt.Sprintf("%x", hash.Sum(nil))
			current[parts[len(parts)-1]] = fileHash
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return metadata, nil
}

// Powered by DeepSeek V3
func compareJSON(data1, data2 map[string]interface{}) map[string]string {
	result := make(map[string]string)

	var flatten func(string, map[string]interface{}, *map[string]string)
	flatten = func(prefix string, m map[string]interface{}, res *map[string]string) {
		for k, v := range m {
			key := prefix + k
			switch val := v.(type) {
			case string:
				(*res)[key] = val
			case map[string]interface{}:
				flatten(key+"/", val, res)
			}
		}
	}

	flat1 := make(map[string]string)
	flat2 := make(map[string]string)
	flatten("", data1, &flat1)
	flatten("", data2, &flat2)

	for k, v2 := range flat2 {
		if v1, exists := flat1[k]; !exists || v1 != v2 {
			result[k] = v2
		}
	}
	return result
}

func CheckOs() string {
	if goos := runtime.GOOS; goos == "windows" {
		return "windows"
	} else if goos == "linux" {
		return "ubuntu"
	} else if goos == "darwin" {
		return "macos"
	}
	return ""
}

func GetRemoteMetadata(goos string, version string) map[string]interface{} {
	resp, err := http.Get("https://update.ravi.cool/metadata?os=" + goos + "&version=" + version)
	if err != nil {
		fmt.Println("Error:", err)
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		fmt.Println("Error: ", resp.Status)
		return nil
	}
	var remoteMetadata map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&remoteMetadata)
	if err != nil {
		fmt.Println("Error:", err)
		return nil
	}
	metadata, ok := remoteMetadata["metadata"].(map[string]interface{})
	if !ok {
		fmt.Println("Error: metadata is not a map")
		return nil
	}
	return metadata
}

func GetDownloadLink(md5hash string) string {
	resp, err := http.Get("https://update.ravi.cool/download/" + md5hash)
	if err != nil {
		fmt.Println("Error:", err)
		return ""
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		fmt.Println("Error: ", resp.Status)
		return ""
	}
	var data map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&data)
	if err != nil {
		fmt.Println("Error:", err)
		return ""
	}
	link, ok := data["download_link"].(string)
	if !ok {
		fmt.Println("Error: link is not a string")
		return ""
	}
	return link
}

func Update(CurrentVersion string) {
	NewVersion, _ := CheckUpdate()
	v1, err := version.NewVersion(NewVersion)
	if err != nil {
		fmt.Println("Error:", err)
	}
	v2, err := version.NewVersion(CurrentVersion)
	if err != nil {
		fmt.Println("Error:", err)
	}
	if v1.LessThan(v2) {
		fmt.Println("Not updating, current version is newer")
	}
	localMetadata, err := GenerateMetadata("./")
	if err != nil {
		fmt.Println("Error:", err)
	}
	file1, _ := os.Create("local.json")
	defer file1.Close()
	encoder1 := json.NewEncoder(file1)
	encoder1.SetIndent("", "  ")
	err = encoder1.Encode(localMetadata)
	if err != nil {
		fmt.Println("Error:", err)
	}
	goos := CheckOs()
	remoteMetadata := GetRemoteMetadata(goos, NewVersion)
	result := compareJSON(localMetadata, remoteMetadata)
	for k, v := range result {
		fmt.Println("File:", k, "Hash:", v)
		link := GetDownloadLink(v)
		if link == "" {
			fmt.Println("Error: link is empty")
			return
		}
		fmt.Println("Download link:", link)
	}
}

func main() {
	content, err := os.ReadFile("version")
	if err != nil {
		fmt.Println("Error:", err)
	}
	v := string(content)
	if v == "" {
		fmt.Println("Error: version is empty")
		return
	} else {
		Update(v)
	}
}
